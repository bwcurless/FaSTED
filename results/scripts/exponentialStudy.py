import json
from decimal import Decimal
import ctypes
from dataclasses import dataclass
import numpy as np
import time
import sys
import math


# Define my own print method so I can tell what output comes from C++, and what from Python
def print(*args, prefix="[PYTHON]", **kwargs):
    __builtins__.print(prefix, *args, **kwargs)


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


class mmaShape(ctypes.Structure):
    _fields_ = [
        ("m", ctypes.c_int),
        ("n", ctypes.c_int),
        ("k", ctypes.c_int),
    ]

    def __repr__(self):
        return f"mmaShape(m={self.m}, n={self.n}, k={self.k})"


# The result types
class Results(ctypes.Structure):
    _fields_ = [
        ("TFLOPS", ctypes.c_double),
        ("pairsFound", ctypes.c_ulonglong),
        ("pairsStored", ctypes.c_ulonglong),
        ("inputProblemShape", mmaShape),
        ("paddedProblemShape", mmaShape),
    ]

    def __repr__(self):
        return f"Results(TFLOPS={self.TFLOPS}, pairsFound={self.pairsFound}, pairsStored={self.pairsStored}, inputProblemShape={self.inputProblemShape}, paddedProblemShape={self.paddedProblemShape})"


# Load the shared library
findPairs = ctypes.CDLL("../../source/main.so")

# Define the function prototype
findPairs.runFromExponentialDataset.restype = Results
findPairs.runFromExponentialDataset.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_bool,
]


def nthroot(a, n):
    return np.power(a, (1 / n))


# Compute the gamma function divisor for determining hyper-sphere volume. Use lgamma so we don't overflow.
def computeGamma(numDim):
    return Decimal(math.lgamma(numDim / 2.0 + 1)).exp()


# returns a new epsilon that should be tested
def adjustEpsilon(oldEpsilon, oldSelectivity, targetSelectivity, numDim):
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
        gammaDec = computeGamma(numDim)

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
def computeSelectivity(resultSetSize, datasetSize):
    return max(0, resultSetSize * 1.0 - datasetSize * 1.0) / datasetSize * 1.0


# Encapsulates the parameters to modify the distribution
class ExponentialDistribution:
    def __init__(self, eLambda, eRange):
        self.eLambda = eLambda
        self.eRange = eRange


# Wraps up the result parameters for an experiment
@dataclass
class Experiment:
    selectivity: float
    epsilon: float
    iteration: int
    results: Results


# Determines the proper epsilon value to obtain a specified selectivity
def findEpsilon(size, dim, selectivity, expD, initialEpsilon=0.1):
    # Selectivity threshold. Must be this % to the target selectivity
    selThreshold = 0.1
    result = findPairs.runFromExponentialDataset(
        size, dim, expD.eLambda, expD.eRange, initialEpsilon, True
    )
    epsilon = initialEpsilon
    while (
        abs((sel := computeSelectivity(result.pairsFound, size)) - selectivity)
        / selectivity
        > selThreshold
    ):
        epsilon = adjustEpsilon(epsilon, sel, selectivity, dim)
        print(f"Selectivity was {sel}/{selectivity} new epsilon: {epsilon}")
        start = time.perf_counter()
        result = findPairs.runFromExponentialDataset(
            size, dim, expD.eLambda, expD.eRange, epsilon, True
        )
        end = time.perf_counter()
        print(f"CUDA Code execution time: {end - start:.6f} seconds")
    print(
        f"Final selectivity was {sel}/{selectivity} with an epsilon of: {epsilon}"
    )
    return epsilon


# Run an experiment over a given range of selectivities.
def runSelectivityExperiment(size, dim, selectivities, expD, iterations=3):
    # First find the appropriate epsilons
    epsilons = {}
    for selectivity in selectivities:
        epsilons[selectivity] = findEpsilon(size, dim, selectivity, expD, 0.1)

    # Now run the actual experiments and save the results
    results = []
    for sel, eps in epsilons.items():
        for i in range(iterations):
            results.append(
                Experiment(
                    sel,
                    eps,
                    i,
                    findPairs.runFromExponentialDataset(
                        size, dim, expD.eLambda, expD.eRange, eps, True
                    ),
                )
            )
    print(results)
    return results


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


# Test how changing the selectivity effects the speed of my algorithm
def runSelectivityVsSpeedExperiment(targetSelectivities, expD):
    print("Running basic selectivity experiment")
    # Run a basic experiment to show that increasing selectivity doesn't significantly effect results
    results = runSelectivityExperiment(1000000, 64, targetSelectivities, expD)
    with open("selectivityVsSpeed.json", "w") as f:
        json.dump(results, f, cls=ResultsEncoder)


# Test how different dataset sizes and dimensionality effect the speed of my algorithm
def runSpeedSweepsExponentialDataExperiment(expD):
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

    # Targeting selectivities in the range of 10..1000
    targetSelectivities = np.logspace(1, 3, 20)

    # Set up my exponential dataset distribution
    expD = ExponentialDistribution(1.0, 5)

    # runSelectivityVsSpeedExperiment(targetSelectivities, expD)

    runSpeedSweepsExponentialDataExperiment(expD)

    # Run on real world datasets. Autotune to use the 3x different selectivities. Will have 3x however many datasets I am testing on of output Pair data.
