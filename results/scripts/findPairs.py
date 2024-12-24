"""
A wrapper that loads the CUDA shared library and makes it available to be run.
"""

import ctypes


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


# Load shared library from file.
def load_findpairs():
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
    return findPairs
