"""
A wrapper that loads the CUDA shared library and makes it available to be run.
"""

import ctypes


def compute_selectivity(result_set_size: int, dataset_size: int) -> float:
    """computes the selectivity as |R|-|D|/|D|

    :result_set_size: How many pairs were found.
    :dataset_size: How many points are in the dataset.

    :Returns: The selectivity.

    """

    return (
        max(0, result_set_size * 1.0 - dataset_size * 1.0) / dataset_size * 1.0
    )


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
        return f"""Results(TFLOPS={self.TFLOPS}, pairsFound={self.pairsFound},
        pairsStored={self.pairsStored}, inputProblemShape={self.inputProblemShape},
        paddedProblemShape={self.paddedProblemShape})"""

    def get_selectivity(self):
        """Calculates and returns the selectivity of the results."""
        return compute_selectivity(self.pairsFound, self.inputProblemShape.m)


# Load shared library from file.
def load_findpairs():
    # Load the shared library
    find_pairs = ctypes.CDLL("../../source/main.so")

    # Define functions
    # Run from a dynamically generated input dataset
    find_pairs.runFromExponentialDataset.restype = Results
    find_pairs.runFromExponentialDataset.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_bool,
    ]

    # Run from a dataset saved to a file
    find_pairs.runFromFile.restype = Results
    find_pairs.runFromFile.argtypes = [
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_bool,
    ]

    # Can re-run with previously allocated input data, and a new epsilon.
    find_pairs.reRun.restype = Results
    find_pairs.reRun.argtypes = [
        ctypes.c_double,
        ctypes.c_bool,
    ]

    # Free all resources allocated on GPU after done running.
    find_pairs.releaseResources.argtypes = []

    return find_pairs
