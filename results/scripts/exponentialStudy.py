import json
from decimal import Decimal
from dataclasses import dataclass
import numpy as np
import time
import sys
import math

from findPairs import mmaShape, Results


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
    eLambda: float
    eRange: float


# Custom encoder to serialize out experiment results
# Extends the default encoder to handle my types
class ResultsEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ctypes.Structure):
            # Convert ctypes.Structure to a dictionary
            return {field[0]: getattr(obj, field[0]) for field in obj._fields_}
        elif isinstance(obj, Experiment):
            # Convert Experiment object to a dictionary
            return obj.__dict__
        return super().default(obj)


class ExperimentRunner(object):
    """Class for running experiments on a pair finding routine."""

    def __init__(self, find_pairs):
        """Constructs an object to perform studies given a pair finding routine.

        :find_pairs: The pair finding class. Can inject in a real one or a mocked one

        """
        self._find_pairs = find_pairs

    # Compute the gamma function divisor for determining hyper-sphere volume. Use lgamma so we don't overflow.
    def computeGamma(self, numDim):
        return Decimal(math.lgamma(numDim / 2.0 + 1)).exp()

    # Performs a binary search to hone in the epsilon until we achieve a target selectivity
    def adjustEpsilonBinary(
        self, lastEpsilon, lastStep, targetSelectivity, lastSelectivity
    ):
        # Ideally at first we just increment by factors of 10 until we find a reasonable epsilon

        # Once we have found it, we binary search that delta to find the exact value.
        if lastSelectivity < 10:
            newEpsilon = lastEpsilon * (10.0 * lastStep)

        return newEpsilon

    # Returns a scaled epsilon based on scaling the volume by how much our selectivity differs from the target selectivity.
    def adjustEpsilonVolume(
        self, oldEpsilon, oldSelectivity, targetSelectivity, numDim
    ):
        # edge case
        # if the selectivity is 0, then we adjust the old selectivity
        # otherwise we'll have a division by 0 error
        if oldSelectivity == 0:
            print(
                "The last selectivity yielded no neighbors with a selectivity of 0; increasing the epsilon value"
            )
            newEpsilon = oldEpsilon * 2.0

        else:
            # Cast to decimal so we don't overflow when we exponentiate
            oldEpsilonDec = Decimal(oldEpsilon)
            targetSelectivityDec = Decimal(targetSelectivity)
            oldSelectivityDec = Decimal(oldSelectivity)
            numDimDec = Decimal(numDim)

            piDec = Decimal(math.pi)
            dec2 = Decimal(2.0)
            gammaDec = self.computeGamma(numDim)

            oldVolumeDec = ((piDec ** (numDimDec / dec2)) / gammaDec) * (
                oldEpsilonDec**numDimDec
            )
            targetVolumeDec = (
                targetSelectivityDec / oldSelectivityDec
            ) * oldVolumeDec

            exprDec = (targetVolumeDec * gammaDec) / (piDec ** (numDim / dec2))
            newEpsilon = float(exprDec ** (Decimal(1.0) / numDimDec))

        return newEpsilon

    # computes the selectivity as |R|-|D|/|D|
    def computeSelectivity(self, resultSetSize, datasetSize):
        return (
            max(0, resultSetSize * 1.0 - datasetSize * 1.0) / datasetSize * 1.0
        )

    # Determines the proper epsilon value to obtain a specified selectivity
    def findEpsilon(self, size, dim, selectivity, expD, initialEpsilon=0.1):
        # Selectivity threshold. Must be this % to the target selectivity
        selThreshold = 0.1
        result = self._find_pairs.runFromExponentialDataset(
            size, dim, expD.eLambda, expD.eRange, initialEpsilon, True
        )
        epsilon = initialEpsilon
        while (
            abs(
                (sel := self.computeSelectivity(result.pairsFound, size))
                - selectivity
            )
            / selectivity
            > selThreshold
        ):
            epsilon = self.adjustEpsilonVolume(epsilon, sel, selectivity, dim)
            print(
                f"Selectivity was {sel}/{selectivity} new epsilon: {epsilon}"
            )
            start = time.perf_counter()
            result = self._find_pairs.runFromExponentialDataset(
                size, dim, expD.eLambda, expD.eRange, epsilon, True
            )
            end = time.perf_counter()
            print(f"CUDA Code execution time: {end - start:.6f} seconds")
        print(
            f"Final selectivity was {sel}/{selectivity} with an epsilon of: {epsilon}"
        )
        return epsilon

    # Run an experiment over a given range of selectivities.
    def runSelectivityExperiment(
        self, size, dim, selectivities, expD, iterations=3
    ):
        # First find the appropriate epsilons
        epsilons = {}
        for selectivity in selectivities:
            epsilons[selectivity] = self.findEpsilon(
                size, dim, selectivity, expD, 0.1
            )

        # Now run the actual experiments and save the results
        results = []
        for sel, eps in epsilons.items():
            for i in range(iterations):
                results.append(
                    Experiment(
                        sel,
                        eps,
                        i,
                        self._find_pairs.runFromExponentialDataset(
                            size, dim, expD.eLambda, expD.eRange, eps, True
                        ),
                    )
                )
        print(results)
        return results

    # Test how changing the selectivity effects the speed of my algorithm
    def runSelectivityVsSpeedExperiment(self, targetSelectivities, expD):
        print("Running basic selectivity experiment")
        # Run a basic experiment to show that increasing selectivity doesn't significantly effect results
        results = runSelectivityExperiment(
            1000000, 64, targetSelectivities, expD
        )
        with open("selectivityVsSpeed.json", "w") as f:
            json.dump(results, f, cls=ResultsEncoder)

    # Test how different dataset sizes and dimensionality effect the speed of my algorithm
    def runSpeedSweepsExponentialDataExperiment(self, expD):
        # Run main speed experiment on exponential datasets
        # Show how problem sizes impacts tensor core utilization
        # Use a fixed selectivity
        selectivity = [10]
        results = []
        print("Running exponential sweep speed experiment")
        for size in np.logspace(3, 6, 20):
            for dim in range(64, 4096, 64):
                results += runSelectivityExperiment(
                    round(size), round(dim), selectivity, expD
                )

        with open("ExpoDataSpeedVsSize.json", "w") as f:
            json.dump(results, f, cls=ResultsEncoder)


if __name__ == "__main__":
    print(sys.version)

    # Create the pair finding routine
    realFindPairs = findPairs.load_findpairs()

    experimentRunner = ExperimentRunner(realFindPairs)

    # Targeting selectivities in the range of 10..1000
    targetSelectivities = np.logspace(1, 3, 20)

    # Set up my exponential dataset distribution
    expD = ExponentialDistribution(1.0, 5)

    # experimentRunner.runSelectivityVsSpeedExperiment(targetSelectivities, expD)

    experimentRunner.runSpeedSweepsExponentialDataExperiment(expD)

    # Run on real world datasets. Autotune to use the 3x different selectivities. Will have 3x however many datasets I am testing on of output Pair data.
