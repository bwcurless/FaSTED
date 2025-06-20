import json
import slurm
import re
import logging
import pathlib
from pathlib import Path
import compute_accuracy
from collections.abc import Callable
import os
import logging
import argparse


def parse_mptc_line(line: str) -> tuple[int, int]:
    m = re.search(r"^(\d+), (\d+)$", line)
    if m:
        query_s, cand_s = m.group(1, 2)
        query, cand = int(query_s), int(cand_s)

        return query, cand
    else:
        raise Exception(f"Failed to parse line from mptc_join: {line}")


def get_mptc_guesses_for_point(mptc_file, point_index: int) -> set[int]:
    """
    Have to read through current position of mptc_file line by line to build up
    the set of guesses. Backtrack if we read too far.
    """
    # Build set for candidate guesses
    guessed_neighbors = set()
    while True:
        # Save old position
        mptc_file_pos = mptc_file.tell()  # Save line in case we need to backtrack

        line = mptc_file.readline()
        # Handle last line in file
        if not line:
            logging.debug(
                f"Guessed neighbors for point {point_index} was\n{guessed_neighbors}"
            )
            return guessed_neighbors

        query, cand = parse_mptc_line(line)
        if query < point_index:
            # Somehow we are still on an old point?? should never happen
            raise Exception("Somehow we seem to have skipped a point")
        elif query == point_index:
            guessed_neighbors.add(cand)
        else:
            # Seek back since we overshot
            mptc_file.seek(mptc_file_pos)
            logging.debug(
                f"Guessed neighbors for point {point_index} was\n{guessed_neighbors}"
            )
            return guessed_neighbors


def parse_gds_line(line: str) -> tuple[int, set[int]]:

    m = re.search(r"^point id: (\d+), neighbors: (.+),$", line)
    if m:
        point_index_string, neighbor_string = m.group(1, 2)
        point_index = int(point_index_string)

        split_neighbor_strings = neighbor_string.split(",")
        expected_neighbors = set([int(x) for x in split_neighbor_strings])
        logging.debug(f"Expected neighbors: {expected_neighbors}")

        return point_index, expected_neighbors
    else:
        raise Exception("Failed to read line from gds_join: {line}")


def point_by_point_averaged_comparison(mptc_path, gds_path):
    """
    The final metric reported in the paper. Compute accuracy of each point,
    then average the accuracy overall points
    """
    running_accuracy_sum = 0.0
    total_points = 0

    with open(mptc_path, "r") as mptc_file, open(gds_path, "r") as gds_file:
        for gds_line in gds_file:
            # Skip intro lines in gds-join file
            if not gds_line.startswith("point id:"):
                continue
            total_points += 1  # Each valid line in GDS file is a point in the dataset
            point_index, expected_neighbors = parse_gds_line(gds_line)
            logging.debug(f"Processing point: {point_index}")

            guesses = get_mptc_guesses_for_point(mptc_file, point_index)
            intersection = len(expected_neighbors.intersection(guesses))
            union = len(expected_neighbors.union(guesses))
            point_score = intersection / union
            logging.debug(f"Current point_score is: {point_score}")

            running_accuracy_sum += point_score

    final_accuracy = running_accuracy_sum / total_points
    return final_accuracy


def global_iou_comparison(left_path, right_path):
    """
    This is the original metric I calculated where I didn't consider point by point,
    all neighors were treated equal.
    """
    # This way computes the accuracy by taking the sum of all intersections over sum of all unions
    input_data = compute_accuracy.create_input_data_from_files(left_path, right_path)
    pair_comparison = compute_accuracy.compute_pair_stats(input_data)
    return pair_comparison


def compare_neighbor_tables(
    base_path: str,
    neighbor_tables: list[tuple[str, str]],
    compare_func: Callable[[Path, Path], object],
):
    """
    Iterate through pairs of files, executing a compare function on each pair and saving
    the results.
    """

    results = {}

    for left_file, right_file in neighbor_tables:
        print(f"Comparing file: {left_file} with file: {right_file}")

        left_path = pathlib.Path(base_path, left_file)
        right_path = pathlib.Path(base_path, right_file)

        pair_comparison = compare_func(left_path, right_path)

        results[f"{left_file}, {right_file}"] = pair_comparison

        print("Comparison done")
        print(f"Results: {results}")

    print("Final comparison results")
    print(results)

    with open("neighbor_table_comparison_results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loglevel", help="Set log level")
    args = parser.parse_args()

    log_level = args.loglevel or os.getenv("LOGLEVEL", "INFO")
    logging.basicConfig(
        level=log_level.upper(), format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info(f"Log level is {log_level}")

    logging.debug("Debug text")

    if not slurm.running_on_slurm():
        # Temporary override for local testing
        base_path = "../accuracy_results/cifar_data/"
        # Local comparisons. Can't copy all the data over, it is far too big.
        neighbor_tables = [
            (
                "mptc_neighbortable_cifar60k_unscaled_0.628906.out",
                "gds_join_neighbortable_FP64_cifar60k_eps_0.62890625.out",
            ),
        ]
        compare_neighbor_tables(
            base_path,
            neighbor_tables,
            point_by_point_averaged_comparison,
        )
    # Real Data
    else:
        base_path = "/scratch/bc2497/pairsData/neighbor_tables"

        neighbor_tables = [
            (
                "fasted/second_run/cifar60k_unscaled_0.628906.pairs",
                "fp64_gds_join/neighbortable_FP64_cifar60k_eps_0.62890625.out",
            ),
            (
                "fasted/second_run/cifar60k_unscaled_0.659180.pairs",
                "fp64_gds_join/neighbortable_FP64_cifar60k_eps_0.6591796875.out",
            ),
            (
                "fasted/second_run/cifar60k_unscaled_0.691406.pairs",
                "fp64_gds_join/neighbortable_FP64_cifar60k_eps_0.69140625.out",
            ),
            (
                "fasted/second_run/gist_unscaled_0.473633.pairs",
                "fp64_gds_join/neighbortable_FP64_gist_eps_0.4736328125.out",
            ),
            (
                "fasted/second_run/gist_unscaled_0.529297.pairs",
                "fp64_gds_join/neighbortable_FP64_gist_eps_0.529296875.out",
            ),
            (
                "fasted/second_run/gist_unscaled_0.593750.pairs",
                "fp64_gds_join/neighbortable_FP64_gist_eps_0.59375.out",
            ),
            (
                "fasted/second_run/sift10m_unscaled_122.500000.pairs",
                "fp64_gds_join/neighbortable_FP64_sift10m_eps_122.5.out",
            ),
            (
                "fasted/second_run/sift10m_unscaled_136.500000.pairs",
                "fp64_gds_join/neighbortable_FP64_sift10m_eps_136.5.out",
            ),
            (
                "fasted/second_run/tiny5m_unscaled_0.183105.pairs",
                "fp64_gds_join/neighbortable_FP64_tiny5m_eps_0.18310546875.out",
            ),
            (
                "fasted/second_run/tiny5m_unscaled_0.204590.pairs",
                "fp64_gds_join/neighbortable_FP64_tiny5m_eps_0.20458984375.out",
            ),
            (
                "fasted/second_run/tiny5m_unscaled_0.227539.pairs",
                "fp64_gds_join/neighbortable_FP64_tiny5m_eps_0.2275390625.out",
            ),
        ]

        compare_neighbor_tables(
            base_path, neighbor_tables, point_by_point_averaged_comparison
        )

        # Old way of computing accuracy
        # compare_neighbor_tables(point_comparisons, global_iou_comparison)
