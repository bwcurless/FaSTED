"""
File: experiments.py
Author: Brian Curless
Description: Contains methods to be automatically determine epsilon values using different
methods like volumetric scaling, and a binary search. Once a certain target selectivity
has been found, experiments are run to obtain the performance of the algorithm.
"""

import ctypes
import json
import math
from pathlib import Path
import time
from typing import Callable, Tuple

from decimal import Decimal
from dataclasses import dataclass
import numpy as np

from experiment.find_pairs import Results, RerunnablePairsFinder


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


def generate_unique_filename(prefix: str, file_type: str) -> str:
    """Insert the current time into the filename to create a unique name"""
    return f"{prefix}_{int(time.time())}{file_type}"


def save_json_results(filename: str, results: dict):
    """Serialize the results to a json file"""
    with open(
        generate_unique_filename(filename, ".json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, cls=ResultsEncoder)


@dataclass
class SearchParameters:
    """Encapsulates the inputs to the search routine"""

    target_selectivity: float
    epsilon: float
    save_pairs: bool


@dataclass
class Experiment:
    """Wraps up the result parameters for an experiment"""

    selectivity: float
    epsilon: float
    iteration: int
    results: Results


def within_percent(actual_value: float, target_value: float, percent: float) -> bool:
    """Checks if an actual value is within a certain percentage of the target value.

    :actual_value: The measured value.
    :target_value: The desired
    :percent: The percentage we want the actual value to be within the target.

    :Returns: True if actual_value is close enough to the target
    """

    return (abs(actual_value - target_value) / target_value) < percent


def compute_gamma(gamma_in: int) -> Decimal:
    """Compute the gamma function divisor for determining hyper-sphere volume.
     Use lgamma so we don't overflow.

    :gamma_in: The value to input to the gamma function.
    :returns: The gamma function result given the input.

    """
    return Decimal(math.lgamma(gamma_in / 2.0 + 1)).exp()


def adjust_epsilon_volume(
    epsilon: float,
    selectivity: float,
    target_selectivity: float,
    num_dim: int,
) -> float:
    """Returns a scaled epsilon based on scaling the volume by how much our selectivity
         differs from the target selectivity. Performs all calculation in Decimal to
        avoid overflowing with high dimensional datasets.

    :epsilon: The current epsilon value.
    :selectivity: The current selectivity.
    :target_selectivity: The desired selectivity value.
    :num_dim: The dimensional of the input data.

    :Returns: The predicted new epsilon value.

    """

    # edge case
    # if the selectivity is 0, then we adjust the old selectivity
    # otherwise we'll have a division by 0 error
    if selectivity == 0:
        print(
            """The last selectivity yielded no neighbors with a selectivity of 0; 
            increasing the epsilon value"""
        )
        new_epsilon = epsilon * 2.0

    else:
        # Cast to decimal so we don't overflow when we exponentiate
        old_epsilon_dec = Decimal(epsilon)
        target_selectivity_dec = Decimal(target_selectivity)
        old_selectivity_dec = Decimal(selectivity)
        num_dim_dec = Decimal(num_dim)

        pi_dec = Decimal(math.pi)
        dec2 = Decimal(2.0)
        gamma_dec = compute_gamma(num_dim)

        old_volume_dec = ((pi_dec ** (num_dim_dec / dec2)) / gamma_dec) * (
            old_epsilon_dec**num_dim_dec
        )
        target_volume_dec = (
            target_selectivity_dec / old_selectivity_dec
        ) * old_volume_dec

        expr_dec = (target_volume_dec * gamma_dec) / (pi_dec ** (num_dim / dec2))
        new_epsilon = float(expr_dec ** (Decimal(1.0) / num_dim_dec))

    return new_epsilon


def bound_epsilon(
    initial_epsilon: float,
    max_epsilon: float,
    target_selectivity: float,
    get_selectivity: Callable[[float], float],
) -> Tuple[float, float]:
    """Find an upper and lower bound on epsilon to achieve a target selectivity,
         given a way to calculate the selectivity for a specific epsilon.

    :initial_epsilon: An epsilon to start searching from.
    :max_epsilon: The maximum epsilon before this routine gives up and Raises a ValueError
    :target_selectivity: The desired selectivity to bound.
    :get_selectivity: A delegate to compute the selectivity given an epsilon value

    :Returns: The range of values that we know epsilon must be in between

    """

    epsilon_scale_factor = 2.0

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

        print(f"Checking if Epsilon is in the range ({old_epsilon}, {new_epsilon})")

        if new_epsilon > max_epsilon:
            raise ValueError(
                f"""Epsilon of {new_epsilon} exceed the maximum value of {max_epsilon}.
                Giving up trying to find the upper bound."""
            )

        current_selectivity = get_selectivity(new_epsilon)
    print(f"Epsilon bounded to: {old_epsilon}, {new_epsilon}")
    return (old_epsilon, new_epsilon)


def adjust_epsilon_binary(
    current_epsilon: float,
    last_epsilon: float,
    target_selectivity: float,
    actual_selectivity: float,
) -> float:
    """Adjust the epsilon value using a binary search method based on a target selectivity.
    Based on if we overshot, or undershot the target selectivity, takes a half step in the
    correct direction and returns that new epsilon value.

    :current_epsilon: The current value for epsilon.
    :last_epsilon: The previous value of epsilon.
    :target_selectivity: The target selectivity that we want.
    :actual_selectivity: The current selectivity.

    :Returns: The new predicted epsilon.

    """

    last_step = abs(current_epsilon - last_epsilon)
    # If we overshot the target selectivity, go backwards
    if actual_selectivity > target_selectivity:
        new_epsilon = current_epsilon - (0.5 * last_step)
    # If we undershot the target selectivity, go forwards
    else:
        new_epsilon = current_epsilon + (0.5 * last_step)

    return new_epsilon


def find_epsilon_binary(
    target_selectivity: float,
    get_selectivity: Callable[[float], float],
    initial_epsilon: float = 0.0,
) -> float:
    """Determines the proper epsilon value to obtain a specified selectivity,
    using a binary search method. First finds a bounded epsilon range,
    then binary searches that range until we achieve the target selectivity.
    The initial epsilon MUST be below the target selectivity for this to work.

    :target_selectivity: The desired selectivity.
    :get_selectivity: A delegate that takes in an epsilon value, and returns the selectivity.
    :initial_epsilon: The first epsilon to search with.

    :Returns: The final epsilon value that achieves the target selectivity.

    """

    # Find the upper and lower bounds for epsilon
    lower_epsilon, upper_epsilon = bound_epsilon(
        initial_epsilon, 100000, target_selectivity, get_selectivity
    )

    # Perform a binary search on these bounds. Start in the middle of the two.
    # We know we overshot it as well, so we can pick up where we left off with
    # the binary search.
    new_epsilon = ((upper_epsilon - lower_epsilon) / 2.0) + lower_epsilon
    last_epsilon = upper_epsilon
    selectivity_threshold = 0.01  # Search within a percent of the target selectivity

    iteration = 0
    current_selectivity = get_selectivity(new_epsilon)
    while not within_percent(
        current_selectivity, target_selectivity, selectivity_threshold
    ):
        print(f"Binary search iteration: {iteration}")
        iteration += 1
        new_epsilon, last_epsilon = (
            adjust_epsilon_binary(
                new_epsilon,
                last_epsilon,
                target_selectivity,
                current_selectivity,
            ),
            new_epsilon,
        )
        start = time.perf_counter()

        current_selectivity = get_selectivity(new_epsilon)

        end = time.perf_counter()
        print(f"CUDA Code execution time: {end - start:.6f} seconds")

    return new_epsilon


def find_epsilon_volumetric(
    size: int,
    dim: int,
    target_selectivity: float,
    get_selectivity: Callable[[float], float],
    initial_epsilon: float = 0.1,
):
    """
    Determines the proper epsilon value to obtain a specified selectivity by iteratively
    scaling epsilon based on the volume of the hyper-sphere.

    :size: The number of points in the dataset.
    :dim: The dimensionality of each point in the dataset.
    :target_selectivity: The desired selectivity.
    :get_selectivity: A delegate to compute the selectivity given an epsilon value
    :initial_epsilon: The first epsilon to search with.

    :Returns: The epsilon that achieves the target selectivity.
    """

    # Selectivity threshold. Must be this % to the target selectivity
    selectivity_threshold = 0.1
    result = get_selectivity(initial_epsilon)

    epsilon = initial_epsilon
    while not within_percent(
        (sel := result.get_selectivity()),
        target_selectivity,
        selectivity_threshold,
    ):
        epsilon = adjust_epsilon_volume(epsilon, sel, target_selectivity, dim)
        print(f"Selectivity was {sel}/{target_selectivity} new epsilon: {epsilon}")
        start = time.perf_counter()
        result = get_selectivity(initial_epsilon)
        end = time.perf_counter()
        print(f"CUDA Code execution time: {end - start:.6f} seconds")
    print(
        f"Final selectivity was {sel}/{target_selectivity} with an epsilon of: {epsilon}"
    )
    return epsilon


# Custom encoder to serialize out experiment results
# Extends the default encoder to handle my types
class ResultsEncoder(json.JSONEncoder):
    """Encodes the results from CUDA into valid json"""

    def default(self, obj):
        if isinstance(obj, ctypes.Structure):
            # Convert ctypes.Structure to a dictionary
            return {field[0]: getattr(obj, field[0]) for field in obj._fields_}
        if isinstance(obj, Experiment):
            # Convert Experiment object to a dictionary
            return obj.__dict__
        return super().default(obj)


class ExperimentRunner:
    """Class for running experiments on a pair finding routine with different dataset types."""

    def __init__(self, find_pairs):
        """Constructs an object to perform studies given a pair finding routine.

        :find_pairs: The pair finding class. Can inject in a real one or a mocked one

        """
        self._find_pairs = find_pairs

    def get_selectivity_rerun(self, epsilon: float) -> float:
        """Obtain the current selecitivity, given an epsilon, by rerunning the pair
        finding routine without changing the dataset.

        :epsilon: The new epsilon value to compute the selectivity for

        :Returns: The selectivity

        """

        results = self._find_pairs.reRun(epsilon, False)
        return results.get_selectivity()

    def run_time_trials(
        self,
        find_pairs: Callable[[float, bool], Results],
        search_params: list[SearchParameters],
        iterations: int,
    ) -> list[Results]:
        """Run find pairs routine for n-iterations and return the results."""
        results = []
        for param in search_params:
            save_pairs = param.save_pairs
            for i in range(iterations):
                # Only save pairs on the first iteration.
                if i > 0:
                    save_pairs = False

                results.append(
                    Experiment(
                        param.target_selectivity,
                        param.epsilon,
                        i,
                        find_pairs(param.epsilon, save_pairs),
                    )
                )
        print(results)
        return results

    def run_selectivity_experiment(
        self,
        selectivities: list[float],
        find_pairs: Callable[[float, bool], Results],
        iterations: int = 3,
        save_pairs: bool = False,
    ) -> Experiment:
        """Run an experiment over a given range of selectivities.
        Auto-finds the epsilon to achieve the target selectivity,
        and repeats the experiment for "iterations" times. Optionally, will save the pairs
        that it finds.

        :selectivities: The selectivities to test over.
        :find_pairs: A delegate to run the algorithm and return the results.
        given an epsilon value.
        :iterations: The number of iterations to run the final tests for.
        :save_pairs: Whether to save the resulting pairs once we have found epsilon. Will only save
        them one time.

        :Returns: The results of the experiments

        """

        # Find the epsilons to achieve the selectivities
        search_params = []
        for selectivity in selectivities:
            search_params.append(
                SearchParameters(
                    selectivity,
                    find_epsilon_binary(
                        selectivity,
                        lambda epsilon: find_pairs(epsilon, False).get_selectivity(),
                    ),
                    save_pairs,
                ),
            )

        # Now run the actual experiments and save the results
        results = self.run_time_trials(find_pairs, search_params, iterations)
        return results

    def build_file_pairs_finder(
        self, base_path: str, dataset: str
    ) -> Callable[[float, bool], Results]:
        """Builds a pairs finder that reads a dataset from file"""

        pairs_finder = lambda epsilon, save_pairs: self._find_pairs.runFromFile(
            str(Path(base_path) / dataset).encode("utf-8"),
            epsilon,
            save_pairs,
        )
        return pairs_finder

    def build_rerunnable_file_pairs_finder(
        self, base_path: str, dataset: str
    ) -> Callable[[float, bool], Results]:
        """Builds a rerunnable pairs finder that reads a dataset from file. Subsequent
        iterations run faster, as the dataset is not read and downloaded to the GPU again.
        This cannot be used for time trialing as the results will be shorter on
        subsequent iterations."""

        pairs_finder = RerunnablePairsFinder(
            lambda epsilon, save_pairs: self._find_pairs.runFromFile(
                str(Path(base_path) / dataset).encode("utf-8"),
                epsilon,
                save_pairs,
            ),
            self._find_pairs.reRun,
        )
        return pairs_finder

    def run_real_dataset_known_epsilons_experiments(
        self,
        dataset_path: str,
        dataset: str,
        search_params: list[SearchParameters],
        num_repetitions: int = 3,
    ) -> None:
        """Runs experiments for a number of iterations on a real world dataset with
        a known epsilon value. Saves the results out to a file"""

        print(f"Running on dataset: {dataset}")
        # Don't rerun here, just load data from scratch so time trials are correct.
        pairs_finder = self.build_file_pairs_finder(dataset_path, dataset)

        results = {}
        results = self.run_time_trials(pairs_finder, search_params, num_repetitions)

        save_json_results(f"{dataset}_results", results)

    def run_real_datasets_epsilon_finder_experiments(
        self, target_selectivities: list[float]
    ) -> None:
        """Finds epsilon value for a list of target selectivities. Saves the results
        out to a file so real experiments can be run with the proper epsilons.

        :target_selectivites: The selectivities to find the epsilons for.

        """
        print("Running on real world datasets")

        # Run on all real world datasets, and find epsilons for each of the target selectivites.
        base_path = Path("/scratch/bc2497/datasets")
        datasets = [
            # "bigcross.txt",
            # "sift10m_unscaled.txt",
            "cifar60k_unscaled.txt",
            # "tiny5m_unscaled.txt",
            # "gist_unscaled.txt",
            # "uscensus.txt",
            # "dataset_fixed_len_pts_expo_NDIM_2_pts_2000_SUPEREGO.txt",
            # "dataset_fixed_len_pts_expo_NDIM_2_pts_2000.txt",
        ]

        # Insert a set of results per dataset
        results = {}

        for dataset in datasets:
            print(f"Running on dataset: {dataset}")
            pairs_finder = self.build_rerunnable_file_pairs_finder(base_path, dataset)

            results[dataset] = self.run_selectivity_experiment(
                target_selectivities, pairs_finder, iterations=1
            )

        save_json_results("realDatasets", results)

    def run_selectivity_vs_speed_experiment(
        self, target_selectivities: list[float]
    ) -> None:
        """Test how changing the selectivity effects the speed of my algorithm

        :target_selectivities: The selectivities to run tests with.

        """
        print("Running basic selectivity experiment")
        size = 100000
        dim = 4096
        e_lambda = 40
        e_range = 10

        # Build a rerunnable pair finding algorithm
        pairs_finder = RerunnablePairsFinder(
            lambda epsilon, save_pairs: self._find_pairs.runFromExponentialDataset(
                size, dim, e_lambda, e_range, epsilon, save_pairs
            ),
            self._find_pairs.reRun,
        )

        # Run a basic experiment to show that increasing selectivity doesn't
        # significantly effect results
        # Chose a size that demonstrates the max throughput.
        results = self.run_selectivity_experiment(target_selectivities, pairs_finder)

        save_json_results("selectivityVsSpeed", results)

    def run_speed_sweeps_exponential_data_experiment(self) -> None:
        """Test how different dataset sizes and dimensionality effect the speed of my algorithm
        Run main speed experiment on exponential datasets
        Show how problem sizes impacts tensor core utilization
        Use a fixed epsilon

        """
        e_lambda = 40
        e_range = 10

        epsilon = 0.001
        selectivity = 0
        results = []
        print("Running exponential sweep speed experiment")
        for size in np.logspace(3, 6, 10):
            rounded_size = round(size)
            # Iterate through powers of 2 for the dimensionality.
            first, last = (64, 4096)

            dim = first
            while dim <= last:
                # Build a rerunnable pair finding algorithm
                pairs_finder = RerunnablePairsFinder(
                    lambda epsilon, save_pairs: self._find_pairs.runFromExponentialDataset(
                        rounded_size,
                        dim,
                        e_lambda,
                        e_range,
                        epsilon,
                        save_pairs,
                    ),
                    self._find_pairs.reRun,
                )

                results += self.run_time_trials(
                    pairs_finder,
                    [SearchParameters(selectivity, epsilon, False)],
                    3,
                )
                dim *= 2

        save_json_results("ExpoDataSpeedVsSize", results)
