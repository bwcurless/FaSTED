import slurm
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import math
import shutil
import os
import json
import logging
import argparse
from pathlib import Path
import neighbor_tables as nt
from dataclasses import dataclass, asdict

# Use tex fonts for plots to match paper
plt.rcParams.update(
    {"text.usetex": True, "font.family": "Computer Modern Roman", "font.size": 8}
)


@dataclass
class ErrorStatistics:
    total_distances: int
    mean_error: float
    max_error: float
    min_error: float


DISTANCE_ERRORS_PREFIX = "distance_errors"
STATISTICS_PREFIX = "stats"


@dataclass
class DistancePaths:
    fasted: Path
    gds: Path
    errors: Path


def compute_distance_error_histogram(fasted_path, gds_path):
    errors_path = nt.prefix_filename(fasted_path, DISTANCE_ERRORS_PREFIX, "txt")

    paths = DistancePaths(fasted_path, gds_path, errors_path)
    stats = compute_distance_errors(paths)

    # Read the file
    with open(errors_path) as f:
        data = [float(line.strip()) for line in f if line.strip()]

    mean = stats.mean_error
    std = np.std(data)
    print(f"Standard deviation is: {std}")

    # Only plot cifar data, not enough room in paper.
    if "cifar" in str(fasted_path):
        # Plot histogram
        plt.figure(figsize=(3.0, 2.0))

        # Bin width
        bin_width = 0.000005

        data_min, data_max = stats.min_error, stats.max_error

        # Extend range to cover whole bins
        left_edge = np.floor((data_min - bin_width / 2) / bin_width) * bin_width
        right_edge = np.ceil((data_max + bin_width / 2) / bin_width) * bin_width

        # Generate bin edges so that 0 is in the center of a bin
        bin_edges = np.arange(left_edge + bin_width / 2, right_edge, bin_width)
        bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)  # Add final edge
        plt.hist(
            data,
            bins=bin_edges,
            color="black",
            edgecolor="black",
            alpha=0.7,
        )
        plt.xlabel(
            "Distance Error",
            fontsize=8,
        )
        plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        plt.gca().ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        plt.ylim(0, 400000)
        plt.xlim(-0.00015, 0.00015)
        plt.xticks(
            horizontalalignment="right",
            rotation_mode="anchor",
            rotation=45,
        )
        plt.ylabel("Frequency", fontsize=8)
        plt.tight_layout()
        plt.savefig("Cifar_distance_error.pdf")
        plt.show()


def get_optimized_file_paths(paths: DistancePaths) -> DistancePaths:
    # Copy files to local storage to speed this up.
    if slurm.running_on_slurm():
        # tmpdir = slurm.get_node_tempdir()
        tmpdir = Path("/tmp/")
        tmp_fasted_path = tmpdir / paths.fasted.name
        tmp_gds_path = tmpdir / paths.gds.name
        print(f"Copying data to temporary storage on node")
        shutil.copyfile(paths.fasted, tmp_fasted_path)
        shutil.copyfile(paths.gds, tmp_gds_path)
        # Declare output data on local storage
        tmp_errors_path = tmpdir / paths.errors.name
        return DistancePaths(tmp_fasted_path, tmp_gds_path, tmp_errors_path)
    else:
        print(f"Working straight from the filesystem.")
        return paths


def compute_distance_errors(original_paths: DistancePaths) -> ErrorStatistics:
    """
    Computes the errors in the distance calculations between
    the two datasets. Saves the results out to errors_path and stats_path.
    """
    stats_path = nt.prefix_filename(original_paths.fasted, STATISTICS_PREFIX, "json")
    # If these files have already been processed, do nothing
    if original_paths.errors.exists() and stats_path.exists():
        print(f'FasTED data "{original_paths.fasted}" has already been processed')
        with open(stats_path, "r") as stats_file:
            json_data = json.load(stats_file)

        return ErrorStatistics(**json_data)

    optimized_paths = get_optimized_file_paths(original_paths)
    print("Computing distance errors.")

    with open(optimized_paths.fasted, "r") as fasted_file, open(
        optimized_paths.gds, "r"
    ) as gds_file, open(optimized_paths.errors, "w") as error_file:
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
                        # If I took the square root of a slightly negative number
                        # It should have been rounded up to 0.0
                        fp16_32_dist = (
                            0.0 if math.isnan(neighbor.distance) else neighbor.distance
                        )
                        fp64_dist = fp64_neighbors[neighbor.point_index]
                        fp64_dist = 0.0 if math.isnan(fp64_dist) else fp64_dist
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

    if slurm.running_on_slurm():
        # Copy error data back to scratch
        print("Copying data off of node.")
        shutil.copyfile(optimized_paths.errors, original_paths.errors)
        # Clean up temporary files
        print("Deleting temporary files.")
        os.remove(optimized_paths.fasted)
        os.remove(optimized_paths.gds)
        os.remove(optimized_paths.errors)

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
        base_path = "../distance_results/"
        # Local comparisons. Can't copy all the data over, it is far too big.
        neighbor_tables_with_distances = [
            (
                "cifar60k_unscaled_0.628906.pairs",
                "neighbortable_distances_FP64_cifar60k_eps_0.62890625.out",
            ),
            (
                "gist_unscaled_0.473633.pairs",
                "neighbortable_distances_FP64_gist_eps_0.4736328125.out",
            ),
            (
                "sift10m_unscaled_122.500000.pairs",
                "neighbortable_distances_FP64_sift10m_eps_122.5.out",
            ),
            (
                "tiny5m_unscaled_0.183105.pairs",
                "neighbortable_distances_FP64_tiny5m_eps_0.18310546875.out",
            ),
        ]
        nt.compare_neighbor_tables(
            base_path,
            neighbor_tables_with_distances,
            compute_distance_error_histogram,
            True,
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

        # Force rerun every time.
        nt.compare_neighbor_tables(
            base_path,
            neighbor_tables_with_distances,
            compute_distance_error_histogram,
            True,
        )
