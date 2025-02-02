import unittest
import numpy as np

import results.scripts.compute_accuracy as compute_accuracy
from results.scripts.compute_accuracy import InputData, PairsStats

def create_test_array() -> np.ndarray:
    return np.array([[1, 1],[2, 2],[3, 3],[4, 4], [5, 5] ])

class AccuracyTests(unittest.TestCase):
    def setUp(self):
        self.left_data = create_test_array()
        self.right_data = create_test_array()

    def build_pair_data(self):
        self.pair_data = InputData(self.left_data, self.right_data)
        
        
    def test_perfect_match(self):
        self.build_pair_data()

        results = compute_accuracy.compute_pair_stats(self.pair_data)

        left_len = self.left_data.shape[0]
        right_len = self.right_data.shape[0]
        expected = PairsStats(left_len, 0, 0, left_len, right_len)
        self.assertEqual(results, expected, f"\nExpected: {expected}\nActual:   {results}")

    def test_left_only_pair_beginning(self):
        extra_pairs = 1
        self.right_data = np.delete(self.right_data, 0, axis=0)
        self.build_pair_data()

        results = compute_accuracy.compute_pair_stats(self.pair_data)

        left_len = self.left_data.shape[0]
        right_len = self.right_data.shape[0]
        expected = PairsStats(left_len - extra_pairs, 0, extra_pairs, left_len, right_len)
        self.assertEqual(results, expected, f"\nExpected: {expected}\nActual:   {results}")

    def test_left_only_pair_middle(self):
        extra_pairs = 1
        self.right_data = np.delete(self.right_data, 2, axis=0)
        self.build_pair_data()

        results = compute_accuracy.compute_pair_stats(self.pair_data)

        left_len = self.left_data.shape[0]
        right_len = self.right_data.shape[0]
        expected = PairsStats(left_len - extra_pairs, 0, extra_pairs, left_len, right_len)
        self.assertEqual(results, expected, f"\nExpected: {expected}\nActual:   {results}")
        
        
   
    def test_left_only_pair_end(self):
        extra_pairs = 1
        self.right_data = np.delete(self.right_data, self.right_data.shape[0]-1, axis=0)
        self.build_pair_data()

        results = compute_accuracy.compute_pair_stats(self.pair_data)

        left_len = self.left_data.shape[0]
        right_len = self.right_data.shape[0]
        expected = PairsStats(left_len - extra_pairs, 0, extra_pairs, left_len, right_len)
        self.assertEqual(results, expected, f"\nExpected: {expected}\nActual:   {results}")

    def test_many_left_only_pairs_end(self):
        extra_pairs = 3
        right_len = self.right_data.shape[0]
        self.right_data = np.delete(self.right_data, range(right_len - extra_pairs, right_len), axis=0)
        self.build_pair_data()

        results = compute_accuracy.compute_pair_stats(self.pair_data)

        left_len = self.left_data.shape[0]
        right_len = self.right_data.shape[0]
        expected = PairsStats(left_len - extra_pairs, 0, extra_pairs, left_len, right_len)
        self.assertEqual(results, expected, f"\nExpected: {expected}\nActual:   {results}")

    def test_right_only_pair_beginning(self):
        missing_pairs = 1
        self.left_data = np.delete(self.left_data, 0, axis=0)
        self.build_pair_data()

        results = compute_accuracy.compute_pair_stats(self.pair_data)

        left_len = self.left_data.shape[0]
        right_len = self.right_data.shape[0]
        expected = PairsStats(left_len, missing_pairs, 0, left_len, right_len)
        self.assertEqual(results, expected, f"\nExpected: {expected}\nActual:   {results}")

    def test_right_only_pair_middle(self):
        missing_pairs = 1
        self.left_data = np.delete(self.left_data, 2, axis=0)
        self.build_pair_data()

        results = compute_accuracy.compute_pair_stats(self.pair_data)

        left_len = self.left_data.shape[0]
        right_len = self.right_data.shape[0]
        expected = PairsStats(left_len, missing_pairs, 0, left_len, right_len)
        self.assertEqual(results, expected, f"\nExpected: {expected}\nActual:   {results}")
        
        
   
    def test_right_only_pair_end(self):
        missing_pairs = 1
        self.left_data = np.delete(self.left_data, self.left_data.shape[0]-1, axis=0)
        self.build_pair_data()

        results = compute_accuracy.compute_pair_stats(self.pair_data)

        left_len = self.left_data.shape[0]
        right_len = self.right_data.shape[0]
        expected = PairsStats(left_len, missing_pairs, 0, left_len, right_len)
        self.assertEqual(results, expected, f"\nExpected: {expected}\nActual:   {results}")

if __name__ == "__main__":
    tests = AccuracyTests()
    tests.setUp()
    tests.test_right_only_pair_end()
