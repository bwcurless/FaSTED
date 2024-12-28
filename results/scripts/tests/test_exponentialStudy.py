import unittest

# Import the module to test
from timeStudies import exponentialStudy

from timeStudies.exponentialStudy import ExperimentRunner
from timeStudies.findPairs import Results, mmaShape


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

    def test_experiment_runner(self):
        print("Running experiment runner")

        results = self.sut.run_selectivity_experiment(100, 10, [10])
        print(results)
        self.assertEqual(True, True)

    def test_selectivity_vs_speed_experiment(self):
        self.sut.run_selectivity_vs_speed_experiment([1, 10, 100])

    def test_size_vs_dim_experiment(self):
        self.sut.run_speed_sweeps_exponential_data_experiment()

    # Pass in a high dimensionality to make sure the method does not error our.
    def test_adjustEpsilonVolume_forOverflow(self):
        old_epsilon = 100
        result = self.sut.adjust_epsilon_volume(old_epsilon, 1000, 10, 10000)
        print(f"Result was: {result}")
        self.assertAlmostEqual(result, old_epsilon, 1)

    def test_boundEpsilon_whenSelectivityEqualsEpsilon(self):
        target_selectivity = 1243
        lower, upper = self.sut.bound_epsilon(
            0.001, 100000, target_selectivity, lambda eps: eps
        )
        self.assertTrue(lower < target_selectivity)
        self.assertTrue(upper > target_selectivity)

    def test_findEpsilonBinary_whenSelectivityEqualsEpsilon(self):
        target_selectivity = 1243
        eps = self.sut.find_epsilon_binary(100, 100, target_selectivity)
        print(
            f"Expected selectivity: {target_selectivity}. Actual selectivity: {eps}"
        )
        self.assertTrue(
            exponentialStudy.within_percent(eps, target_selectivity, 0.01)
        )


if __name__ == "__main__":
    unittest.main()
