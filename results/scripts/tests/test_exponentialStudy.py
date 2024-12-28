import unittest

# Import the module to test
from timeStudies import exponentialStudy

from timeStudies.exponentialStudy import ExperimentRunner
from timeStudies.exponentialStudy import ExponentialDistribution
from timeStudies.findPairs import Results, mmaShape


# Fake pair finding class for testing.
class MockFindPairs:
    def __init__(self):
        self.__last_size = 0
        self.__last_dim = 0

    def runFromExponentialDataset(
        self, size, dim, e_lambda, e_range, epsilon, skip_pairs
    ):
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
        self.expD = ExponentialDistribution(1, 2)

    def test_experiment_runner(self):
        print("Running experiment runner")

        results = self.sut.runSelectivityExperiment(100, 10, [10], self.expD)
        print(results)
        self.assertEqual(True, True)

    # Pass in a high dimensionality to make sure the method does not error our.
    def test_adjustEpsilonVolume_forOverflow(self):
        oldEpsilon = 100
        result = self.sut.adjustEpsilonVolume(oldEpsilon, 1000, 10, 10000)
        print(f"Result was: {result}")
        self.assertAlmostEqual(result, oldEpsilon, 1)

    def test_boundEpsilon_whenSelectivityEqualsEpsilon(self):
        target_selectivity = 1243
        lower, upper = self.sut.boundEpsilon(
            0.001, 100000, target_selectivity, lambda eps: eps
        )
        self.assertTrue(lower < target_selectivity)
        self.assertTrue(upper > target_selectivity)

    def test_findEpsilonBinary_whenSelectivityEqualsEpsilon(self):
        target_selectivity = 1243
        eps = self.sut.findEpsilonBinary(
            100, 100, target_selectivity, self.expD
        )
        print(
            f"Expected selectivity: {target_selectivity}. Actual selectivity: {eps}"
        )
        self.assertTrue(
            exponentialStudy.withinPercent(eps, target_selectivity, 0.01)
        )


if __name__ == "__main__":
    unittest.main()
