import json
import ctypes
import numpy as np
import sys
import math


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
findPairs = ctypes.CDLL("./main.so")

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


# returns a new epsilon that should be tested
def adjustEpsilon(oldEpsilon, oldSelectivity, targetSelectivity, numDim):

    # edge case
    # if the selectivity is 0, then we adjust the old selectivity
    # otherwise we'll have a division by 0 error
    if oldSelectivity == 0:
        print(
            "[Python] The last selectivity yielded no neighbors with a selectivity of 0; doubling the epsilon value"
        )
        newEpsilon = oldEpsilon * 2.0

    else:

        oldVolume = (
            (math.pi ** (numDim / 2.0)) / (math.gamma(numDim / 2.0 + 1))
        ) * (oldEpsilon**numDim)
        targetVolume = (
            targetSelectivity * 1.0 / oldSelectivity * 1.0
        ) * oldVolume

        expr = (targetVolume * (math.gamma(numDim / 2.0 + 1))) / (
            math.pi ** (numDim / 2.0)
        )
        newEpsilon = nthroot(expr, numDim)

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
class Experiment:
    def __init__(self, selectivity, epsilon, iteration, results):
        self.selectivity = selectivity
        self.epsilon = epsilon
        self.iteration = iteration
        self.results = results


# Determines the proper epsilon value to obtain a specified selectivity
def findEpsilon(size, dim, selectivity, expD, initialEpsilon=0.1):
    result = findPairs.runFromExponentialDataset(
        size, dim, expD.eLambda, expD.eRange, initialEpsilon, True
    )
    epsilon = initialEpsilon
    while (
        abs((sel := computeSelectivity(result.pairsFound, size)) - selectivity)
        / selectivity
        > 0.1
    ):
        epsilon = adjustEpsilon(epsilon, sel, selectivity, dim)
        print(f"Selectivity was {sel}/{selectivity} new epsilon: {epsilon}")
        result = findPairs.runFromExponentialDataset(
            size, dim, expD.eLambda, expD.eRange, epsilon, True
        )
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
    for size in np.logspace(1, 6, 20):
        for dim in range(64, 4096, 64):
            results += runSelectivityExperiment(
                round(size), round(dim), selectivity, expD
            )

    with open("ExpoDataSpeedVsSize.json", "w") as f:
        json.dump(results, f, cls=ResultsEncoder)


print(sys.version)

# Targeting selectivities in the range of 10..1000
targetSelectivities = np.logspace(1, 3, 20)

# Set up my exponential dataset distribution
expD = ExponentialDistribution(1.0, 5)

runSelectivityVsSpeedExperiment(targetSelectivities, expD)

# runSpeedSweepsExponentialData(expD)


# Run on real world datasets. Autotune to use the 3x different selectivities. Will have 3x however many datasets I am testing on of output Pair data.
