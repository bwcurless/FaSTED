"""
File: plotResults.py
Description: Plots experimental results for publishing.
"""

import json
import os
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Map out column names for later
SEL_COL = "selectivity"
SPEED_COL = "results.TFLOPS"
INPUT_SIZE_COL_M = "results.inputProblemShape.m"
INPUT_SIZE_COL_N = "results.inputProblemShape.n"
INPUT_DIM_COL = "results.inputProblemShape.k"
TIME = "results.totalTime"
ALGORITHM = "algorithm"
DATASET_NAME = "dataset"

# Algorithms
MPTC_JOIN = "MPTC-Join"
GDS_JOIN = "GDS-Join"
TED_JOIN = "TED-Join"


def parse_selectivity_vs_speed_data():
    """Read in the selectivity vs speed data and create plots for it"""

    with open(
        "../speed_results/selectivityVsSpeed.json", "r", encoding="utf-8"
    ) as f:
        selec_speed_results = json.load(f)
    # Pretty print json
    # print(json.dumps(selec_speed_results, indent=4))

    df = pd.json_normalize(selec_speed_results)
    # Average results from each iteration
    averaged_results = (
        df.groupby(SEL_COL, as_index=False).mean().drop(columns=["iteration"])
    )
    print(averaged_results)
    plt.figure(1)
    plt.plot(
        averaged_results[SEL_COL], averaged_results[SPEED_COL], marker="."
    )
    plt.xlabel("Selectivity Level")
    plt.ylabel("TFLOPS")
    plt.ylim(0, 300)
    plt.savefig("selectivityVsSpeed.pdf")
    plt.show()


def parse_speed_vs_size_data():
    """Read in the speed vs size data and create plots for it."""

    with open(
        "../speed_results/ExpoDataSpeedVsSize_OnDemandRasterizer.json",
        "r",
        encoding="utf-8",
    ) as f:
        speed_size_results = json.load(f)
    df = pd.json_normalize(speed_size_results)
    # Average results from each iteration. We used a fixed selectity here, so groupBy input shape.
    averaged_results = (
        df.groupby([INPUT_SIZE_COL_M, INPUT_DIM_COL], as_index=False)
        .mean()
        .drop(columns=["iteration"])
    )
    print(averaged_results)

    # Extract x and y labels...
    unique_size = np.unique(averaged_results[INPUT_SIZE_COL_M])
    unique_dim = np.unique(averaged_results[INPUT_DIM_COL])

    num_sizes_tested = len(unique_size)
    num_dims_tested = len(unique_dim)

    max_speed = averaged_results[SPEED_COL].max()

    # Convert results to 2D array to show as a heatmap
    grid = np.zeros((num_dims_tested, num_sizes_tested))

    for dim, size, speed in zip(
        averaged_results[INPUT_DIM_COL],
        averaged_results[INPUT_SIZE_COL_M],
        averaged_results[SPEED_COL],
    ):
        dim_index = np.where(unique_dim == dim)[0]
        size_index = np.where(unique_size == size)[0]
        grid[dim_index, size_index] = speed

    # Create the figure
    fig, ax = plt.subplots()
    cax = ax.imshow(
        grid.T,
        cmap="viridis",
        origin="lower",
        aspect="auto",
        vmin=0,
        vmax=160,
        extent=(
            0,
            num_dims_tested,
            0,
            num_sizes_tested,
        ),
    )

    # To center labels on pixels, must offset by a half a pixel
    center_offset = 0.5

    ax.set_xlabel("Dataset Dimensionlity ($d$)")
    ax.set_xticks(np.arange(num_dims_tested) + center_offset)
    ax.set_xticklabels(unique_dim)

    ax.set_ylabel("Dataset Size ($|D|$)")
    ax.set_yticks(np.arange(num_sizes_tested) + center_offset)
    ax.set_yticklabels(unique_size)

    # Add speed labels for each cell
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = int(grid[i, j])
            ax.text(
                i + center_offset,
                j + center_offset,
                value,
                ha="center",
                va="center",
                color="white" if value < 0.5 * max_speed else "black",
                fontsize=10,
            )

    colorbar = fig.colorbar(cax, ax=ax)
    colorbar.set_label("Throughput (TFLOPS)")

    # Get the current ticks (positions) of the colorbar
    ticks = colorbar.get_ticks()
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{tick:.2f}" for tick in ticks])

    plt.tight_layout()
    plt.savefig("ExpoDataSpeedVsSize.pdf")
    plt.show()


def plot_real_world_data_speed_comparison():
    """Plots performance comparison between algorithms on real world
    datasets"""

    def extract_mptc_results(filename, dataset_name):
        with open(
            os.path.join("../speed_results/", filename),
            "r",
            encoding="utf-8",
        ) as f:
            json_results = json.load(f)
        raw_results = pd.json_normalize(json_results)
        # Average results from each iteration.
        averaged_results = (
            raw_results.groupby([SEL_COL], as_index=False)
            .mean()
            .drop(columns=["iteration"])
        )

        averaged_results[DATASET_NAME] = dataset_name
        # Mark these results as my own
        averaged_results[ALGORITHM] = MPTC_JOIN
        # Drop extraneous columns
        minimal_results = averaged_results[
            [ALGORITHM, DATASET_NAME, TIME, SEL_COL]
        ]

        return minimal_results

    # Assemble all mptc results
    dataset_labels = [
        "cifar60k",
        "tiny5m",
        "sift10m",
        "gist1m",
    ]
    cifar = dataset_labels[0]
    tiny = dataset_labels[1]
    sift = dataset_labels[2]
    gist = dataset_labels[3]

    mptc_files = [
        "cifar60k_unscaled.txt_results_1737956025.json",
        "gist_unscaled.txt_results_1738039369.json",
        "sift10m_unscaled.txt_results_1738048428.json",
        "tiny5m_unscaled.txt_results_1737960896.json",
    ]

    all_results = pd.DataFrame()
    for file, dataset_name in zip(mptc_files, dataset_labels):
        all_results = pd.concat(
            [all_results, extract_mptc_results(file, dataset_name)],
            ignore_index=True,
        )

    # Assemble all gds-join results
    def add_row(
        df: pd.DataFrame,
        algorithm: str,
        dataset: str,
        time: float,
        selectivity: int,
    ) -> None:
        new_row = {
            ALGORITHM: algorithm,
            DATASET_NAME: dataset,
            TIME: time,
            SEL_COL: selectivity,
        }
        df.loc[len(df)] = new_row

    add_row(all_results, GDS_JOIN, cifar, 2.329744667, 64)
    add_row(all_results, GDS_JOIN, cifar, 2.675126, 128)
    add_row(all_results, GDS_JOIN, cifar, 3.067051667, 256)

    add_row(all_results, GDS_JOIN, tiny, 670.0995993, 64)
    add_row(all_results, GDS_JOIN, tiny, 835.6346603, 128)
    add_row(all_results, GDS_JOIN, tiny, 1056.39112, 256)

    add_row(all_results, GDS_JOIN, sift, 3392.178252, 64)
    add_row(all_results, GDS_JOIN, sift, 4279.934975, 128)
    add_row(all_results, GDS_JOIN, sift, 5195.027983, 256)

    add_row(all_results, GDS_JOIN, gist, 272.1477887, 64)
    add_row(all_results, GDS_JOIN, gist, 359.6358573, 128)
    add_row(all_results, GDS_JOIN, gist, 444.5647683, 256)

    print(all_results)

    # TODO Assemble all ted-join results

    # Create the figure
    algorithms = [MPTC_JOIN, GDS_JOIN]

    # Create 4 subplots, one for each dataset
    plot_rows = 1
    plot_cols = 4
    fig, ax = plt.subplots(
        plot_rows, plot_cols, layout="constrained", figsize=(10, 3)
    )

    # Choose a colormap (e.g., "viridis", "plasma", "Greys", etc.)
    colormap = cm.get_cmap("Greys")

    # Normalize the colors to map them to the colormap
    colors = dict(
        zip(algorithms, colormap(np.linspace(0.5, 1, len(algorithms))))
    )

    # Collect all handles and labels from each axis to create only one legend
    handles, labels = [], []

    # Hardcode y limits specific to data for better readability
    y_limits = [
        4,
        1250,
        6000,
        500,
    ]

    figure_label_prefixes = [
        "(a)",
        "(b)",
        "(c)",
        "(d)",
    ]

    # Create all subplots
    for i, axis in enumerate(ax.flat):
        dataset_name = dataset_labels[i]
        single_dataset = all_results[all_results[DATASET_NAME] == dataset_name]
        single_dataset_minimal_columns = single_dataset[
            [ALGORITHM, TIME, SEL_COL]
        ].sort_values(by=SEL_COL, ascending=True)

        selectivities = np.unique(single_dataset_minimal_columns[SEL_COL])
        collapsed_data = (
            single_dataset_minimal_columns.groupby(ALGORITHM)
            .agg(
                {
                    TIME: list,
                }
            )
            .reset_index()
        )

        # Reorder so my algorithm appears first on charts
        reordered_collapsed_data = collapsed_data.sort_values(
            by=ALGORITHM, ascending=False
        )

        x = np.arange(len(selectivities))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0
        for algorithm, times in zip(
            reordered_collapsed_data[ALGORITHM],
            reordered_collapsed_data[TIME],
        ):
            offset = width * multiplier
            axis.bar(
                x + offset,
                times,
                width,
                label=algorithm,
                color=colors[algorithm],
            )
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        axis.set_axisbelow(True)
        axis.grid(axis="y")
        axis.set_ylim(0, y_limits[i])
        axis_label_fontsize = 9
        # Only label first y axis to reduce clutter
        if i == 0:
            axis.set_ylabel("Time (s)", fontsize=axis_label_fontsize)
        figure_label = f"{figure_label_prefixes[i]} {dataset_name}"
        axis.set_title(figure_label, y=-0.5, fontsize=8)
        axis.set_xticks(x + width, selectivities)
        axis.set_xlabel("Selectivity", fontsize=axis_label_fontsize)
        # Decrease size of tick labels
        axis.tick_params(axis="both", labelsize=axis_label_fontsize - 1)

        # Save this axis's labels and handles for the high level legend
        h, l = axis.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Remove duplicates by using a dictionary (preserves order)
    unique = dict(zip(labels, handles))

    # Add a single legend for the entire figure with unique labels
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=4,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig("RealDataSetSpeedComparison.pdf")
    plt.show()


def compute_iou():
    """Given the accuracy data, compute the Intersection over the
    union of the pairs."""
    with open(
        "../accuracy_results/real_world_accuracy_data.json", "r"
    ) as file:
        data = json.load(file)
        accuracies = {
            k.split("_")[0]: (
                100
                * v["both_sides"]
                / (v["both_sides"] + v["left_only"] + v["right_only"])
            )
            for (k, v) in data.items()
        }
        print(accuracies)
        # Convert data to a latex table.


if __name__ == "__main__":
    # parse_selectivity_vs_speed_data()
    parse_speed_vs_size_data()
    plot_real_world_data_speed_comparison()
    compute_iou()
