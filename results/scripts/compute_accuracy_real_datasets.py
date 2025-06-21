import slurm
import neighbor_tables as nt
import logging
import compute_accuracy
import os
import logging
import argparse


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
            point_index, expected_neighbors = nt.parse_gds_neighbor_line(gds_line)
            logging.debug(f"Processing point: {point_index}")

            guesses = nt.get_mptc_guesses_for_point(mptc_file, point_index)
            intersection = len(set(expected_neighbors).intersection(guesses))
            union = len(set(expected_neighbors).union(guesses))
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
        nt.compare_neighbor_tables(
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

        nt.compare_neighbor_tables(
            base_path, neighbor_tables, point_by_point_averaged_comparison
        )

        # Old way of computing accuracy
        # compare_neighbor_tables(point_comparisons, global_iou_comparison)
