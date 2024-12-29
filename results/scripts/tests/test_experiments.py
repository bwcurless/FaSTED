"""
File: test_experiments.py
Description: Runs experiments with a mocked pair finding routine to quickly prove out
the python algorithms before running on an actual  GPU.
"""

import unittest

from experiment.experiments import ExperimentRunner
from experiment.find_pairs import Results, mmaShape
from experiment import experiments


class MockFindPairs:
    """Fake pair finding class for testing."""

    def __init__(self):
        self.__last_size = 0
        self.__last_dim = 0

    def runFromExponentialDataset(
        self, size, dim, e_lambda, e_range, epsilon, skip_pairs
    ):
        """Fake method to run from an exponential dataset."""

        # Save the size for when we rerun
        self.__last_size = size
        self.__last_dim = dim
        result = Results()
        result.TFLOPS = 1.0
        # The selectivity is equal to epsilon
        result.pairsFound = int(epsilon * size)
        result.pairsStored = result.pairsFound
        result.inputProblemShape = mmaShape(size, size, dim)
        result.paddedProblemShape = mmaShape(size, size, dim)

        return result

    def reRun(self, epsilon, skip_pairs):
        result = Results()
        result.TFLOPS = 1.0
        # The selectivity is equal to epsilon
        result.pairsFound = int(epsilon * self.__last_size)
        result.pairsStored = result.pairsFound
        result.inputProblemShape = mmaShape(
            self.__last_size, self.__last_size, self.__last_dim
        )
        result.paddedProblemShape = mmaShape(
            self.__last_size, self.__last_size, self.__last_dim
        )

        return result


class TestExponentialStudy(unittest.TestCase):
    def setUp(self):
        # Make a fake pairs finding routine
        fake_findpairs = MockFindPairs()

        self.sut = ExperimentRunner(fake_findpairs)

    def test_selectivity_vs_speed_experiment(self):
        self.sut.run_selectivity_vs_speed_experiment([1, 10, 100])

    def test_size_vs_dim_experiment(self):
        self.sut.run_speed_sweeps_exponential_data_experiment()

    # Pass in a high dimensionality to make sure the method does not error our.
    def test_adjustEpsilonVolume_forOverflow(self):
        old_epsilon = 100
        result = experiments.adjust_epsilon_volume(
            old_epsilon, 1000, 10, 10000
        )
        print(f"Result was: {result}")
        self.assertAlmostEqual(result, old_epsilon, 1)


if __name__ == "__main__":
    unittest.main()
