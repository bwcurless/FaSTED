"""
File: run_experiments.py
Description: Execute the experiments that I want to run.
"""

import sys

from experiment.experiments import ExperimentRunner
from experiment.find_pairs import load_findpairs

if __name__ == "__main__":
    print(sys.version)
    print(sys.path)

    # Create the pair finding routine
    real_find_pairs = load_findpairs()

    experiment_runner = ExperimentRunner(real_find_pairs)

    # Targeting selectivities in the range of 10..1000
    # test_selectivities = np.logspace(1, 3, 20)
    # experiment_runner.run_selectivity_vs_speed_experiment(test_selectivities)

    experiment_runner.run_speed_sweeps_exponential_data_experiment()

    # Run on real world datasets. Autotune to use the 3x different selectivities.
    # Will have 3x however many datasets I am testing on of output Pair data.
    test_selectivities = [10, 100, 1000]
