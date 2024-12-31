"""
A wrapper that loads the CUDA shared library and makes it available to be run.
"""

from typing import Callable
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

    def get_selectivity(self) -> float:
        """Calculates and returns the selectivity of the results."""
        return compute_selectivity(self.pairsFound, self.inputProblemShape.m)


class RerunnablePairsFinder:
    """Wraps a pair finding routine for a specified dataset in a way where it can be quickly
    rerun on subsequent iterations. You only need to read/generate the dataset once, and copy it
    to the GPU once. After that, the algorithm can be rerun quickly by changing epsilon. This class
    encapsulates the state management, and lets any pair finding routine be run and rerun.
    """

    def __init__(
        self,
        first_find_pairs: Callable[[float, bool], Results],
        rerun_find_pairs: Callable[[float, bool], Results],
    ):
        self._first_find_pairs = first_find_pairs
        self._rerun_find_pairs = rerun_find_pairs
        self._first_run = True

    def __call__(self, epsilon: float, save_pairs: bool = False) -> Results:
        """Runs the pair finding routine with a given epsilon. Returns the results.

        :epsilon: The search radius.
        :save_pairs: If the GPU should save the resulting pairs.

        :Returns: The results of the search

        """

        if self._first_run:
            self._first_run = False
            print("First time running find_pairs, running full method")
            return self._first_find_pairs(epsilon, save_pairs)

        print("Rerunning find_pairs")
        return self._rerun_find_pairs(epsilon, save_pairs)


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
