"""
File: test_experiments.py
Description: Runs experiments with a mocked pair finding routine to quickly prove out
the python algorithms before running on an actual  GPU.
"""

import unittest

from experiment.experiments import ExperimentRunner, SearchParameters
from experiment.find_pairs import Results, mmaShape
from experiment import experiments


class MockFindPairs:
    """Fake pair finding class for testing."""

    def __init__(self):
        self._last_size = 0
        self._last_dim = 0

    def runFromExponentialDataset(
        self, size, dim, e_lambda, e_range, epsilon, skip_pairs
    ) -> Results:
        """Fake method to run from an exponential dataset."""

        # Save the size for when we rerun
        self._last_size = size
        self._last_dim = dim
        result = Results()
        result.TFLOPS = 1.0
        # The selectivity is equal to epsilon
        result.pairsFound = int(epsilon * size)
        result.pairsStored = result.pairsFound
        result.inputProblemShape = mmaShape(size, size, dim)
        result.paddedProblemShape = mmaShape(size, size, dim)

        return result

    def runFromFile(self, filename, epsilon, skip_pairs) -> Results:
        """Fake method to run from a file"""
        # Make up a size
        self._last_size = 100000
        self._last_dim = 15
        result = Results()
        result.TFLOPS = 1.0
        # The selectivity is equal to epsilon
        result.pairsFound = int(epsilon * self._last_size)
        result.pairsStored = result.pairsFound
        result.inputProblemShape = mmaShape(
            self._last_size, self._last_size, self._last_dim
        )
        result.paddedProblemShape = mmaShape(
            self._last_size, self._last_size, self._last_dim
        )
        return result

    def reRun(self, epsilon, skip_pairs) -> Results:
        """Fake method to rerun a dataset"""
        result = Results()
        result.TFLOPS = 1.0
        # The selectivity is equal to epsilon
        result.pairsFound = int(epsilon * self._last_size)
        result.pairsStored = result.pairsFound
        result.inputProblemShape = mmaShape(
            self._last_size, self._last_size, self._last_dim
        )
        result.paddedProblemShape = mmaShape(
            self._last_size, self._last_size, self._last_dim
        )

        return result


TEST_SELECTIVITIES = [1, 10, 100]


class TestExponentialStudy(unittest.TestCase):
    def setUp(self):
        # Make a fake pairs finding routine
        fake_findpairs = MockFindPairs()

        self.sut = ExperimentRunner(fake_findpairs)

    def test_selectivity_vs_speed_experiment(self):
        self.sut.run_selectivity_vs_speed_experiment(TEST_SELECTIVITIES)

    def test_size_vs_dim_experiment(self):
        self.sut.run_speed_sweeps_exponential_data_experiment()

    def test_real_datasets_experiment(self):
        self.sut.run_real_datasets_epsilon_finder_experiments(
            TEST_SELECTIVITIES
        )

    def test_run_real_dataset_known_epsilons_experiments(self):
        search_params = [
            SearchParameters(64, 1.2),
            SearchParameters(128, 3.4),
            SearchParameters(256, 5.6),
        ]
        self.sut.run_real_dataset_known_epsilons_experiments(
            "basePath", "dateset", search_params
        )

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
