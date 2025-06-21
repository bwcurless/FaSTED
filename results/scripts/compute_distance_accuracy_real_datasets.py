import slurm
import math
import os
import json
import logging
import argparse
from pathlib import Path
import neighbor_tables as nt
from dataclasses import dataclass, asdict


@dataclass
class ErrorStatistics:
    total_distances: int
    mean_error: float
    max_error: float
    min_error: float


def prefix_filename(filepath: str, prefix: str, extension: str) -> Path:
    stripped_ext = extension.removeprefix(".")
    path = Path(filepath)
    return path.parent / Path(f"{prefix}_{path.stem}.{stripped_ext}")


DISTANCE_ERRORS_PREFIX = "distance_errors"
STATISTICS_PREFIX = "stats"


def compute_distance_error_histogram(fasted_path, gds_path):
    errors_path = prefix_filename(fasted_path, DISTANCE_ERRORS_PREFIX, "txt")

    stats = compute_distance_errors(fasted_path, gds_path, errors_path)
    # TODO compute std dev and histogram


def compute_distance_errors(
    fasted_path, gds_path, errors_path: Path
) -> ErrorStatistics:
    """
    Computes the errors in the distance calculations between
    the two datasets. Saves the results out to errors_path and stats_path.
    """
    stats_path = prefix_filename(fasted_path, STATISTICS_PREFIX, "json")
    # If these files have already been processed, do nothing
    if errors_path.exists() and stats_path.exists():
        print(f'FasTED data "{fasted_path}" has already been processed')
        with open(stats_path, "r") as stats_file:
            json_data = json.load(stats_file)

        return ErrorStatistics(**json_data)

    with open(fasted_path, "r") as fasted_file, open(gds_path, "r") as gds_file, open(
        errors_path, "w"
    ) as error_file:
        total_neighbors = 0
        min = float("inf")
        max = float("-inf")
        sum = 0.0
        while True:
            gds_line = gds_file.readline()

            # Handle last line in file
            if not gds_line:
                break
            # Skip intro lines in gds-join file
            if not gds_line.startswith("point id:"):
                continue

            point_index, fp64_neighbor_indices = nt.parse_gds_neighbor_line(gds_line)

            point_index_2, fp64_neighbors = nt.parse_gds_distance_line(
                gds_file.readline()
            )
            if point_index != point_index_2:
                raise Exception(f"Point indexes don't match for point {point_index}")
            logging.debug(f"Processing point: {point_index}")
            # Create dictionary of distances from fp64 data
            fp64_neighbors = dict(zip(fp64_neighbor_indices, fp64_neighbors))

            fp16_32_neighbors = nt.get_fasted_neighbors_for_point(
                fasted_file, point_index
            )
            # Compute the errors between matched distances
            for neighbor in fp16_32_neighbors:
                if neighbor.point_index in fp64_neighbors:
                    if neighbor.distance is not None:
                        # I took the square root of a slightly negative number
                        # in my code when I computed the distance between identical points.
                        # It should have been rounded up to 0.0
                        fp16_32_dist = (
                            0.0
                            if math.isnan(neighbor.distance)
                            and point_index == neighbor.point_index
                            else neighbor.distance
                        )
                        fp64_dist = fp64_neighbors[neighbor.point_index]
                        dist = fp16_32_dist - fp64_dist
                        if dist > max:
                            max = dist
                        if dist < min:
                            min = dist
                        sum += dist
                        total_neighbors += 1
                        error_file.write(f"{dist}\n")
                    else:
                        raise Exception(
                            f"Distance was not present in FasTED data. This shouldn't happen."
                        )
    json_data = ErrorStatistics(total_neighbors, sum / total_neighbors, max, min)
    print(f"Stats\n{json_data}")
    with open(stats_path, "w") as stats_file:
        json.dump(asdict(json_data), stats_file, indent=2)

    return json_data


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
        base_path = "../distance_results/cifar/"
        # Local comparisons. Can't copy all the data over, it is far too big.
        neighbor_tables_with_distances = [
            (
                "cifar60k_unscaled_0.628906.pairs",
                "neighbortable_distances_FP64_cifar60k_eps_0.62890625.out",
            ),
        ]
        nt.compare_neighbor_tables(
            base_path,
            neighbor_tables_with_distances,
            compute_distance_error_histogram,
        )
    # Real Data
    else:
        base_path = "/scratch/bc2497/pairsData/neighbor_tables_with_distances"

        neighbor_tables_with_distances = [
            (
                "fasted/cifar60k_unscaled_0.628906.pairs",
                "fp64_gds_join/neighbortable_distances_FP64_cifar60k_eps_0.62890625.out",
            ),
            (
                "fasted/gist_unscaled_0.473633.pairs",
                "fp64_gds_join/neighbortable_distances_FP64_gist_eps_0.4736328125.out",
            ),
            (
                "fasted/sift10m_unscaled_122.500000.pairs",
                "fp64_gds_join/neighbortable_distances_FP64_sift10m_eps_122.5.out",
            ),
            (
                "fasted/tiny5m_unscaled_0.183105.pairs",
                "fp64_gds_join/neighbortable_distances_FP64_tiny5m_eps_0.18310546875.out",
            ),
        ]

        nt.compare_neighbor_tables(
            base_path,
            neighbor_tables_with_distances,
            compute_distance_error_histogram,
        )
