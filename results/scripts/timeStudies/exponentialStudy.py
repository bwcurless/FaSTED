"""
File: exponentialStudy.py
Author: Brian Curless
Description: Contains methods to be automatically determine epsilon values using different
methods like volumetric scaling, and a binary search. Once a certain target selectivity has been found, experiments are run to obtain the performance of the algorithm.
"""

import ctypes
import json
import time
import sys
import math

from decimal import Decimal
from dataclasses import dataclass
import numpy as np

from timeStudies.findPairs import load_findpairs
from timeStudies.findPairs import Results


# Define my own print method so I can tell what output comes from C++, and what from Python
# def print(*args, prefix="[PYTHON]", **kwargs):
#    __builtins__.print(prefix, *args, **kwargs)


# A timing decorator to profile some code
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(
            f"Function '{func.__name__}' executed in {end_time - start_time:.6f} seconds"
        )
        return result

    return wrapper


# Wraps up the result parameters for an experiment
@dataclass
class Experiment:
    selectivity: float
    epsilon: float
    iteration: int
    results: Results


# Encapsulates the parameters to modify the distribution
@dataclass
class ExponentialDistribution:
    e_lambda: float
    e_range: float


# Checks if an actual value is within a certain percentage of the target value.
def withinPercent(actualValue, targetValue, percent):
    return (abs(actualValue - targetValue) / targetValue) < percent


# Custom encoder to serialize out experiment results
# Extends the default encoder to handle my types
class ResultsEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ctypes.Structure):
            # Convert ctypes.Structure to a dictionary
            return {field[0]: getattr(obj, field[0]) for field in obj._fields_}
        if isinstance(obj, Experiment):
            # Convert Experiment object to a dictionary
            return obj.__dict__
        return super().default(obj)


class ExperimentRunner:
    """Class for running experiments on a pair finding routine."""

    def __init__(self, find_pairs):
        """Constructs an object to perform studies given a pair finding routine.

        :find_pairs: The pair finding class. Can inject in a real one or a mocked one

        """
        self._find_pairs = find_pairs

    def compute_gamma(self, gamma_in: int) -> Decimal:
        """Compute the gamma function divisor for determining hyper-sphere volume.
         Use lgamma so we don't overflow.

        :gamma_in: The value to input to the gamma function.
        :returns: The gamma function result given the input.

        """
        return Decimal(math.lgamma(gamma_in / 2.0 + 1)).exp()

    # Returns a scaled epsilon based on scaling the volume by how much our selectivity
    # differs from the target selectivity.
    def adjustEpsilonVolume(
        self,
        old_epsilon: float,
        old_selectivity: float,
        target_selectivity: float,
        num_dim: int,
    ):
        # edge case
        # if the selectivity is 0, then we adjust the old selectivity
        # otherwise we'll have a division by 0 error
        if old_selectivity == 0:
            print(
                """The last selectivity yielded no neighbors with a selectivity of 0; 
                increasing the epsilon value"""
            )
            new_epsilon = old_epsilon * 2.0

        else:
            # Cast to decimal so we don't overflow when we exponentiate
            old_epsilon_dec = Decimal(old_epsilon)
            target_selectivity_dec = Decimal(target_selectivity)
            old_selectivity_dec = Decimal(old_selectivity)
            num_dim_dec = Decimal(num_dim)

            pi_dec = Decimal(math.pi)
            dec2 = Decimal(2.0)
            gamma_dec = self.compute_gamma(num_dim)

            old_volume_dec = ((pi_dec ** (num_dim_dec / dec2)) / gamma_dec) * (
                old_epsilon_dec**num_dim_dec
            )
            target_volume_dec = (
                target_selectivity_dec / old_selectivity_dec
            ) * old_volume_dec

            expr_dec = (target_volume_dec * gamma_dec) / (
                pi_dec ** (num_dim / dec2)
            )
            new_epsilon = float(expr_dec ** (Decimal(1.0) / num_dim_dec))

        return new_epsilon

    # computes the selectivity as |R|-|D|/|D|
    def computeSelectivity(self, result_set_size, dataset_size):
        return (
            max(0, result_set_size * 1.0 - dataset_size * 1.0)
            / dataset_size
            * 1.0
        )

    # Find an upper and lower bound on epsilon to achieve a target selectivity,
    # given a way to calculate the selectivity for a specific epsilon.
    def boundEpsilon(
        self, initial_epsilon, max_epsilon, target_selectivity, get_selectivity
    ):
        epsilon_scale_factor = 10.0

        current_selectivity = 0
        new_epsilon = initial_epsilon
        # Do a rough incremental search where we overshoot, so we can bound the search space.
        while current_selectivity < target_selectivity:
            old_epsilon = new_epsilon
            # If we started at 0, scale by addition at first...
            new_epsilon = (
                epsilon_scale_factor
                if (old_epsilon == 0)
                else (old_epsilon * epsilon_scale_factor)
            )

            print(
                f"Checking if Epsilon is in the range ({old_epsilon}, {new_epsilon})"
            )

            if new_epsilon > max_epsilon:
                raise ValueError(
                    f"""Epsilon of {new_epsilon} exceed the maximum value of {max_epsilon}.
                    Giving up trying to find the upper bound."""
                )

            current_selectivity = get_selectivity(new_epsilon)
        print(f"Epsilon bounded to: {old_epsilon}, {new_epsilon}")
        return (old_epsilon, new_epsilon)

    # Adjust the epsilon value using a binary search method based on a target selectivity.
    def adjustEpsilonBinary(
        self,
        current_epsilon,
        last_epsilon,
        target_selectivity,
        actual_selectivity,
    ):
        last_step = abs(current_epsilon - last_epsilon)
        # If we overshot the target selectivity, go backwards
        if actual_selectivity > target_selectivity:
            new_epsilon = current_epsilon - (0.5 * last_step)
        # If we undershot the target selectivity, go forwards
        else:
            new_epsilon = current_epsilon + (0.5 * last_step)

        return new_epsilon

    # Determines the proper epsilon value to obtain a specified selectivity,
    # using a binary search method. First finds a bounded epsilon range,
    # then binary searches that range until we achieve the target selectivity.
    # The initial epsilon MUST be below the target selectivity for this to work.
    def findEpsilonBinary(
        self, size, dim, target_selectivity, exp_d, initial_epsilon=0.0
    ):
        # Create a simple way to get the selectivity given an epsilon
        get_selectivity = lambda epsilon: self.computeSelectivity(
            self._find_pairs.runFromExponentialDataset(
                size, dim, exp_d.e_lambda, exp_d.e_range, epsilon, True
            ).pairsFound,
            size,
        )
        # To rerun the routine without changing the dataset run this method.
        get_selectivity_rerun = lambda epsilon: self.computeSelectivity(
            self._find_pairs.reRun(epsilon, True).pairsFound,
            size,
        )

        # Run once to generate the dataset and push it to the device.
        get_selectivity(1.0)

        # Find the upper and lower bounds for epsilon
        lower_epsilon, upper_epsilon = self.boundEpsilon(
            initial_epsilon, 100000, target_selectivity, get_selectivity_rerun
        )

        # Perform a binary search on these bounds. Start in the middle of the two.
        # We know we overshot it as well, so we can pick up where we left off with the binary search.
        new_epsilon = ((upper_epsilon - lower_epsilon) / 2.0) + lower_epsilon
        last_epsilon = upper_epsilon
        selectivity_threshold = (
            0.01  # Search within a percent of the target selectivity
        )

        iteration = 0
        current_selectivity = get_selectivity_rerun(new_epsilon)
        while not withinPercent(
            current_selectivity, target_selectivity, selectivity_threshold
        ):
            print(f"Binary search iteration: {iteration}")
            iteration += 1
            new_epsilon, last_epsilon = (
                self.adjustEpsilonBinary(
                    new_epsilon,
                    last_epsilon,
                    target_selectivity,
                    current_selectivity,
                ),
                new_epsilon,
            )
            start = time.perf_counter()

            current_selectivity = get_selectivity_rerun(new_epsilon)

            end = time.perf_counter()
            print(f"CUDA Code execution time: {end - start:.6f} seconds")

        return new_epsilon

    # Determines the proper epsilon value to obtain a specified selectivity by iteratively
    # scaling epsilon based on the volume of the hyper-sphere.
    def findEpsilonVolumetric(
        self, size, dim, target_selectivity, exp_d, initial_epsilon=0.1
    ):
        # Selectivity threshold. Must be this % to the target selectivity
        selectivity_threshold = 0.1
        result = self._find_pairs.runFromExponentialDataset(
            size, dim, exp_d.e_lambda, exp_d.e_range, initial_epsilon, True
        )
        epsilon = initial_epsilon
        while not withinPercent(
            (sel := self.computeSelectivity(result.pairsFound, size)),
            target_selectivity,
            selectivity_threshold,
        ):
            epsilon = self.adjustEpsilonVolume(
                epsilon, sel, target_selectivity, dim
            )
            print(
                f"Selectivity was {sel}/{target_selectivity} new epsilon: {epsilon}"
            )
            start = time.perf_counter()
            result = self._find_pairs.runFromExponentialDataset(
                size, dim, exp_d.e_lambda, exp_d.e_range, epsilon, True
            )
            end = time.perf_counter()
            print(f"CUDA Code execution time: {end - start:.6f} seconds")
        print(
            f"Final selectivity was {sel}/{target_selectivity} with an epsilon of: {epsilon}"
        )
        return epsilon

    # Run an experiment over a given range of selectivities.
    def runSelectivityExperiment(
        self, size, dim, selectivities, exp_d, iterations=3
    ):
        # First find the appropriate epsilons
        epsilons = {}
        for selectivity in selectivities:
            epsilons[selectivity] = self.findEpsilonBinary(
                size, dim, selectivity, exp_d, 0.0
            )

        # Now run the actual experiments and save the results
        # There is an assumption that the input data has already been generated and downloaded to
        # the device. Don't repeat it here, just rerun.
        results = []
        for sel, eps in epsilons.items():
            for i in range(iterations):
                results.append(
                    Experiment(
                        sel,
                        eps,
                        i,
                        self._find_pairs.reRun(eps, True),
                    )
                )
        print(results)
        return results

    def run_selectivity_vs_speed_experiment(
        self, target_selectivities: list[float], exp_d: ExponentialDistribution
    ):
        """Test how changing the selectivity effects the speed of my algorithm

        :target_selectivities: The selectivities to run tests with.
        :exp_d: The distribution of the input data.

        """
        print("Running basic selectivity experiment")
        # Run a basic experiment to show that increasing selectivity doesn't
        # significantly effect results
        # Chose a size that demonstrates the max throughput.
        results = self.runSelectivityExperiment(
            100000, 4096, target_selectivities, exp_d
        )
        with open("selectivityVsSpeed.json", "w") as f:
            json.dump(results, f, cls=ResultsEncoder)

    # Test how different dataset sizes and dimensionality effect the speed of my algorithm
    def runSpeedSweepsExponentialDataExperiment(self, exp_d):
        # Run main speed experiment on exponential datasets
        # Show how problem sizes impacts tensor core utilization
        # Use a fixed selectivity
        selectivity = [10]
        results = []
        print("Running exponential sweep speed experiment")
        for size in np.logspace(3, 6, 20):
            for dim in range(64, 4096, 64):
                results += self.runSelectivityExperiment(
                    round(size), round(dim), selectivity, exp_d
                )

        with open("ExpoDataSpeedVsSize.json", "w") as f:
            json.dump(results, f, cls=ResultsEncoder)


if __name__ == "__main__":
    print(sys.version)

    # Create the pair finding routine
    real_find_pairs = load_findpairs()

    experiment_runner = ExperimentRunner(real_find_pairs)

    # Targeting selectivities in the range of 10..1000
    target_selectivities = np.logspace(1, 3, 20)

    # Set up my exponential dataset distribution
    exp_d = ExponentialDistribution(1.0, 5)

    experiment_runner.run_selectivity_vs_speed_experiment(
        target_selectivities, exp_d
    )

    # experiment_runner.runSpeedSweepsExponentialDataExperiment(exp_d)

    # Run on real world datasets. Autotune to use the 3x different selectivities.
    # Will have 3x however many datasets I am testing on of output Pair data.
