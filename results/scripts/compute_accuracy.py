from dataclasses import dataclass
from enum import Enum
import numpy as np

class PairsIterator:
    """Iterator for the input data"""
    def __init__ (self, input_array: np.ndarray):
        self.current_index = 0
        self.array = input_array

    def current_row(self) -> np.ndarray:
        # Don't read past end of array
        index = min(self.total_elements() - 1, self.current_index)
        return self.array[index]

    def increment_row(self) -> None:
        self.current_index += 1

    def has_values_left(self) -> bool:
        return self.current_index < self.total_elements()

    def total_elements(self) -> int:
        return self.array.shape[0]


@dataclass
class InputData:
    def __init__(self, test, truth):
        self.left = PairsIterator(test)
        self.right = PairsIterator(truth)
        
    left: PairsIterator
    right: PairsIterator

@dataclass
class PairsStats:
    both_sides: int = 0
    right_only: int = 0
    left_only: int = 0
    total_test_pairs: int = 0
    total_truth_pairs: int = 0
    

class CompareResult(Enum):
    SAME=0
    BEFORE=1
    AFTER=2

def compare_row(left: np.ndarray, right: np.ndarray) -> CompareResult:
    """ Determine where in the sequence the left row comes relative to the right row"""

    for dim in range(left.shape[0]):
        if left[dim] < right[dim]:
            return CompareResult.BEFORE
        elif left[dim] > right[dim]:
            return CompareResult.AFTER
        # Passing through denotes equivalence (not > or <)
    
    return CompareResult.SAME
    
def create_input_data_from_files(filepath1: str, filepath2: str) -> InputData:
    file1 = np.loadtxt(filepath1, delimiter=',')
    file2 = np.loadtxt(filepath2, delimiter=',')

    input_data = InputData(file1, file2)

    return input_data
    
    
def compute_pair_stats(data: InputData) -> PairsStats:
    results = PairsStats()

    while(data.left.has_values_left() or data.right.has_values_left()):

        compare_result = compare_row(data.left.current_row(), data.right.current_row())
        #print(f"Comparing row: {data.left.current_index} to: {data.right.current_index}")
    
        # The comparisons is all from the perspective of the left set
        match compare_result:
            case CompareResult.SAME:
                # [1, 1] <-> [1, 1]
                results.both_sides += 1
                data.left.increment_row()
                data.right.increment_row()

            # These are trickier, depending on which list has values left to compare
            # changes how to categorize the pair. 
            case CompareResult.BEFORE:
                # [0, 0] <-> [0, 0]
                #   ...       ...
                # [1, 1] <-> [2, 2]
                # [1, 2] <->
                # [1, 3] <->
                if data.left.has_values_left():
                    results.left_only += 1
                    data.left.increment_row()
                # [0, 0] <-> [0, 0]
                #   ...       ...
                # [1, 1] <-> [2, 2]
                #        <-> [2, 3]
                #        <-> [2, 4]
                else:
                    results.right_only += 1
                    data.right.increment_row()

            case CompareResult.AFTER:
                # [0, 0] <-> [0, 0]
                #   ...       ...
                # [1, 1] <-> [0, 0]
                #        <-> [0, 1]
                #        <-> [0, 2]
                if data.right.has_values_left():
                    results.right_only += 1
                    data.right.increment_row()
                # [0, 0] <-> [0, 0]
                #   ...       ...
                # [1, 1] <-> [0, 0]
                # [1, 2] <-> 
                # [1, 3] <-> 
                else:
                    results.left_only += 1
                    data.left.increment_row()

            case _:
                raise RuntimeError


    results.total_test_pairs = data.left.total_elements()
    results.total_truth_pairs = data.right.total_elements()
    return results



if __name__ == "__main__":
    print("Running")
