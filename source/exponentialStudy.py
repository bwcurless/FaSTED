import ctypes
import numpy as np
import sys
import math


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


# Determines the proper epsilon value to obtain a specified selectivity
def findEpsilon(size, dim, selectivity, initialEpsilon=0.01):
    lda = 1
    rang = 10.0
    result = findPairs.runFromExponentialDataset(
        size, dim, lda, rang, initialEpsilon
    )
    epsilon = initialEpsilon
    while (
        abs((sel := computeSelectivity(result.pairsFound, size)) - selectivity)
        > 1
    ):
        epsilon = adjustEpsilon(epsilon, sel, selectivity, dim)
        print(f"Selectivity was {sel}/{selectivity} new epslion: {epsilon}")
        result = findPairs.runFromExponentialDataset(
            size, dim, lda, rang, epsilon
        )
    print(
        f"Final selectivity was {sel}/{selectivity} with an epsilon of: {epsilon}"
    )
    return epsilon


# Run epsilon sweep to see if it effects TFLOPS significantly. Do this on it's own.
def runSelectivitySweep(selectivities):
    epsilons = {}
    for selectivity in selectivities:
        epsilons[selectivity] = findEpsilon(2000, 64, selectivity, 0.01)


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
]

print("Version in script")
print(sys.version)

# Search for epsilons that achieve these selectivities
targetSelectivities = [10]

runSelectivitySweep(targetSelectivities)

# Run scaling sweep.

# Run 3x time trials for each final epsilon.

# Run experimental sweeps. Autotune the selectivity to hit a desired target.
# Input:
# Dataset size
# Dataset dimensionality
# Epsilon

# Output:
# TFLOPS
# #Pairs/Selectivity
#

# Run on real world datasets. Autotune to use the 3x different selectivities. Will have 3x however many datasets I am testing on of output Pair data.
