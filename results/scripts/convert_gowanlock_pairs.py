""" This converts gowanlocks pair data to my format, which is easier to read into
numpy """

import os
import re
from pathlib import Path
import numpy as np


def convert_all_pair_sets():
    base_path = "/scratch/bc2497/pairsData"

    cifar60k_dp_file = (
        "stripped_neighbortable_FP64_cifar60k_eps_0.62890625.txt"
    )
    gist_dp_file = "stripped_neighbortable_FP64_gist_eps_0.4736328125.txt"
    sift10m_dp_file = "stripped_neighbortable_FP64_sift10m_eps_122.5.txt"
    tiny5m_dp_file = "stripped_neighbortable_FP64_tiny5m_eps_0.18310546875.txt"

    convert_pairs(Path(base_path), cifar60k_dp_file)
    convert_pairs(Path(base_path), gist_dp_file)
    convert_pairs(Path(base_path), sift10m_dp_file)
    convert_pairs(Path(base_path), tiny5m_dp_file)


def convert_pairs(base_path: Path, filename1: str):
    print(f"Converting file {filename1}")
    input_filepath = os.path.join(base_path, filename1)
    output_filepath = os.path.join(base_path, "flattened_" + filename1)
    with open(input_filepath, "r") as infile:
        with open(output_filepath, "w") as outfile:
            for row, line in enumerate(infile):
                # if row > 10:
                #    break

                # Extract first id, then extract a list of Id's afterwards
                first_id_pattern = r"^point id: (\d+),"
                pairs_pattern = r"neighbors: (.*)$"
                first_id = re.search(first_id_pattern, line).group(1)
                pairs = (
                    re.search(pairs_pattern, line)
                    .group(1)
                    .rstrip(",")
                    .split(",")
                )

                for pair in pairs:
                    # For each one, create a new pair in an output file
                    outfile.write(f"{first_id},{pair}\n")
    print(f"Done converting file {filename1}")


if __name__ == "__main__":
    convert_all_pair_sets()
