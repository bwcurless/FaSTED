"""
File: run_experiments.py
Description: Execute the experiments that I want to run.
"""

import sys

from experiment.experiments import ExperimentRunner, SearchParameters
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

    # experiment_runner.run_speed_sweeps_exponential_data_experiment()

    # Run on real world datasets. Autotune to use the 3x different selectivities.
    # Will have 3x however many datasets I am testing on of output Pair data.
    test_selectivities = [64, 128, 256]
    # experiment_runner.run_real_datasets_epsilon_finder_experiments(test_selectivities)

    SCRATCH_PATH = "/scratch/bc2497/datasets/cuSimSearch_real_data"

    experiment_runner.run_real_dataset_known_epsilons_experiments(
        SCRATCH_PATH,
        "cifar60k_unscaled.txt",
        [
            SearchParameters(64, 0.62890625, True),
            SearchParameters(128, 0.6591796875, True),
            SearchParameters(256, 0.69140625, True),
        ],
        1,
    )
    experiment_runner.run_real_dataset_known_epsilons_experiments(
        SCRATCH_PATH,
        "tiny5m_unscaled.txt",
        [
            SearchParameters(64, 0.18310546875, True),
            SearchParameters(128, 0.20458984375, True),
            SearchParameters(256, 0.2275390625, True),
        ],
        1,
    )
    experiment_runner.run_real_dataset_known_epsilons_experiments(
        SCRATCH_PATH,
        "gist_unscaled.txt",
        [
            SearchParameters(64, 0.4736328125, True),
            SearchParameters(128, 0.529296875, True),
            SearchParameters(256, 0.59375, True),
        ],
        1,
    )
    experiment_runner.run_real_dataset_known_epsilons_experiments(
        SCRATCH_PATH,
        "sift10m_unscaled.txt",
        [
            SearchParameters(64, 122.5, True),
            SearchParameters(128, 136.5, True),
            SearchParameters(256, 152.5, True),
        ],
        1,
    )
