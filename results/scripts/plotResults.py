"""
File: plotResults.py
Description: Plots experimental results for publishing.
"""

import json
import math
import os
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Use tex fonts for plots to match paper
plt.rcParams.update(
    {"text.usetex": True, "font.family": "Computer Modern Roman", "font.size": 8}
)

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
TED_JOIN = "TED-Join-Index"
MISTIC = "MiSTIC"


def parse_selectivity_vs_speed_data():
    """Read in the selectivity vs speed data and create plots for it"""

    with open("../speed_results/selectivityVsSpeed.json", "r", encoding="utf-8") as f:
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
    plt.plot(averaged_results[SEL_COL], averaged_results[SPEED_COL], marker=".")
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
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
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

    ax.set_xlabel("Dataset Dimensionality ($d$)")
    ax.set_xticks(
        np.arange(num_dims_tested) + center_offset,
        unique_dim,
        horizontalalignment="right",
        rotation=45,
    )

    ax.set_ylabel("Dataset Size ($|D|$)")
    ax.set_yticks(np.arange(num_sizes_tested) + center_offset, unique_size)

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
                fontsize=6,
            )

    colorbar = fig.colorbar(cax, ax=ax)
    colorbar.set_label("Throughput (TFLOPS)")

    # Get the current ticks (positions) of the colorbar
    ticks = colorbar.get_ticks()
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{tick:.0f}" for tick in ticks])

    plt.tight_layout()
    plt.savefig("ExpoDataSpeedVsSize.pdf")
    plt.show()


@dataclass
class SpeedResults:
    """Used to give a json file with speed results a display friendly name"""

    friendly_name: str
    dimensionality: int
    filepath: str


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
        minimal_results = averaged_results[[ALGORITHM, DATASET_NAME, TIME, SEL_COL]]

        return minimal_results

    # Order to go low to high dimensionality sift, tiny, cifar, gist
    dataset_results = [
        SpeedResults("sift10m", 128, "sift10m_unscaled.txt_results_1738048428.json"),
        SpeedResults("tiny5m", 384, "tiny5m_unscaled.txt_results_1737960896.json"),
        SpeedResults("cifar60k", 512, "cifar60k_unscaled.txt_results_1737956025.json"),
        SpeedResults("gist1m", 960, "gist_unscaled.txt_results_1738039369.json"),
    ]

    # Assemble all mptc results
    sift = dataset_results[0].friendly_name
    tiny = dataset_results[1].friendly_name
    cifar = dataset_results[2].friendly_name
    gist = dataset_results[3].friendly_name

    all_results = pd.DataFrame()
    for result in dataset_results:
        file, dataset_name = (result.filepath, result.friendly_name)
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

    # Manually compute the averages here
    # Subtracts out the "Time to estimate batches" from "Time to join"
    tj_tiny_64 = (
        14195.661126
        - 5562.255205
        + 14190.259351
        - 5559.080249
        + 14183.869225
        - 5551.792481
    ) / 3.0

    tj_tiny_128 = (
        16476.701999
        - 5611.305432
        + 16475.117487
        - 5608.609297
        + 16474.570044
        - 5607.122338
    ) / 3.0

    tj_tiny_256 = (
        19298.439721
        - 5603.901025
        + 19316.790174
        - 5620.080564
        + 19314.235564
        - 5619.702950
    ) / 3.0

    tj_sift_64 = (
        10706.653007
        - 2482.418047
        + 10705.409744
        - 2480.845796
        + 10708.060327
        - 2482.671413
    ) / 3.0

    tj_sift_128 = (
        12514.143639
        - 2539.549293
        + 12514.728978
        - 2537.274044
        + 12511.698752
        - 2538.492326
    ) / 3.0
    tj_sift_256 = (
        14447.869670
        - 2628.108349
        + 14436.264242
        - 2619.244253
        + 14439.257269
        - 2621.461369
    ) / 3.0

    add_row(all_results, TED_JOIN, tiny, tj_tiny_64, 64)
    add_row(all_results, TED_JOIN, tiny, tj_tiny_128, 128)
    add_row(all_results, TED_JOIN, tiny, tj_tiny_256, 256)

    add_row(all_results, TED_JOIN, sift, tj_sift_64, 64)
    add_row(all_results, TED_JOIN, sift, tj_sift_128, 128)
    add_row(all_results, TED_JOIN, sift, tj_sift_256, 256)

    # Mistic results
    add_row(all_results, MISTIC, cifar, 4.765354, 64)
    add_row(all_results, MISTIC, cifar, 5.046123, 128)
    add_row(all_results, MISTIC, cifar, 6.246613, 256)

    add_row(all_results, MISTIC, tiny, 662.925975, 64)
    add_row(all_results, MISTIC, tiny, 987.669580, 128)
    add_row(all_results, MISTIC, tiny, 1418.637501, 256)

    add_row(all_results, MISTIC, sift, 2169.987598, 64)
    add_row(all_results, MISTIC, sift, 2517.037860, 128)
    add_row(all_results, MISTIC, sift, 2798.847250, 256)

    add_row(all_results, MISTIC, gist, 209.062901, 64)
    add_row(all_results, MISTIC, gist, 285.383825, 128)
    add_row(all_results, MISTIC, gist, 385.351438, 256)

    print(all_results)

    # Create the figure
    algorithms = [MPTC_JOIN, GDS_JOIN, TED_JOIN, MISTIC]

    # Create 4 subplots, one for each dataset
    plot_rows = 1
    plot_cols = 4
    fig, ax = plt.subplots(
        plot_rows, plot_cols, layout="constrained", figsize=(7.25, 3)
    )

    # Scientific notation to save space
    for axis in ax:
        axis.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    hatches = ["////", "\\\\\\\\", "oooo", "xxxx"]  # , "+", "x", "o", "O", ".", "*"]

    # Map the hatches to algorithms
    hatch_map = dict(zip(algorithms, [hatches[i] for i in range(len(algorithms))]))

    # Collect all handles and labels from each axis to create only one legend
    handles, labels = [], []

    # Hardcode y limits specific to data for better readability
    # These are based on the original order, should maybe reorder these earlier
    y_limits = [
        15000,
        15000,
        8,
        600,
    ]

    figure_label_prefixes = [
        "(a)",
        "(b)",
        "(c)",
        "(d)",
    ]

    # Create all subplots
    for i, axis in enumerate(ax.flat):
        dataset_name = dataset_results[i].friendly_name
        dimensionality = dataset_results[i].dimensionality
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
        algo_order = [MPTC_JOIN, MISTIC, GDS_JOIN, TED_JOIN]
        reordered_collapsed_data = collapsed_data.sort_values(
            by=ALGORITHM,
            key=lambda col: col.map({name: i for i, name in enumerate(algo_order)}),
        )

        x = np.arange(len(selectivities))  # the label locations
        width = 0.20  # the width of the bars
        multiplier = 0

        def getTimes(algorithm: str):
            return reordered_collapsed_data.loc[
                reordered_collapsed_data[ALGORITHM] == algorithm, TIME
            ].iloc[0]

        # Find mptc times to compute speedup
        mptc_times = getTimes(MPTC_JOIN)
        print(f"MPTC times: {mptc_times}")

        for algorithm, times in zip(
            reordered_collapsed_data[ALGORITHM],
            reordered_collapsed_data[TIME],
        ):
            offset = width * multiplier

            rects = axis.bar(
                x + offset,
                times,
                width,
                label=algorithm,
                fill=False,
                hatch=hatch_map[algorithm],
            )

            def round_sig(x, sig=2):
                if x == 0:
                    return "0"
                return f"{round(x, sig - int(math.floor(math.log10(abs(x)))) - 1):g}"

            # Put speedup labels in for quick reference
            if algorithm != MPTC_JOIN:
                this_algo_times = getTimes(algorithm)
                speedups = [
                    round_sig(x / y) for x, y in zip(this_algo_times, mptc_times)
                ]
                axis.bar_label(
                    rects, labels=speedups, rotation=90, fontsize=6, padding=2
                )

            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        axis.set_ylim(0, y_limits[i])
        axis.set_axisbelow(True)
        # Only label first y axis to reduce clutter
        if i == 0:
            axis.set_ylabel("Time (s)")
        figure_label = (
            f"{figure_label_prefixes[i]} {dataset_name} $(d = {dimensionality} )$"
        )
        axis.set_title(figure_label, y=-0.5)
        axis.set_xticks(x + width, selectivities)
        axis.set_xlabel("Selectivity")
        # Decrease size of tick labels
        # axis.tick_params(axis="both", labelsize=axis_label_fontsize - 1)

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
    with open("../accuracy_results/real_world_accuracy_data.json", "r") as file:
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


def synthetic_flops_comparison():
    # All experiments were run with a 100,000 point dataset
    dataset_size = 100000

    def compute_tflops(time: float, dims: int):
        return dataset_size**2 * dims / (time * 1e12)

    mptc_dims = [64, 128, 256, 512, 1024, 2048, 4096]
    mptc_join = [17, 30, 55, 91, 132, 148, 154]
    tj_dims = [64, 128, 256, 384]
    ted_join_brute_times = [0.96, 2.43, 13.43, 20.12]  # Only was able to run up to 384D
    ted_join_brute_flops = [
        compute_tflops(time, dims)
        for (time, dims) in zip(ted_join_brute_times, tj_dims)
    ]

    # Max performance lines
    fp16_fp32_max = 312  # Top dashed red line
    fp64_max = 19.5  # Middle dashed green line

    fig, ax = plt.subplots(figsize=(3.5, 2.25))

    # Dashed max throughput lines
    ax.axhline(y=fp16_fp32_max, color="red", linestyle="--", label="TC FP16-FP32 Max")
    offset_coords = (0, 2.5)
    max_flops_label_x_pos = 4.5
    ax.annotate(
        "312 TFLOPS",
        (max_flops_label_x_pos, 312),
        xytext=offset_coords,
        textcoords="offset points",
    )
    ax.axhline(y=fp64_max, color="green", linestyle="--", label="TC FP64 Max")
    ax.annotate(
        "19.5 TFLOPS",
        (max_flops_label_x_pos, 19.5),
        xytext=offset_coords,
        textcoords="offset points",
    )

    mptc_indices = range(len(mptc_dims))
    ax.set_xticks(mptc_indices, [str(x) for x in mptc_dims])

    # Plot MPTC-Join
    ax.plot(mptc_indices, mptc_join, color="red", marker=".", label="MPTC-Join")

    # Plot TED-Join Brute
    ax.plot(
        [0, 1, 2, 2.5],  # Half indice because 384 is halfway between my ticks
        ted_join_brute_flops,
        color="green",
        marker="s",
        markersize="3",
        label="TED-Join-Brute",
    )

    # Log scale for y-axis
    ax.set_yscale("log")
    ax.set_ylim(0.1, 10**3)

    # Labels
    ax.set_xlabel("Dataset Dimensionality ($d$)")
    ax.set_ylabel("Throughput (TFLOPS)")

    # Grid, legend, and layout
    # ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig("synthetic_flops_comparison.pdf")
    plt.show()


if __name__ == "__main__":
    # parse_selectivity_vs_speed_data()
    parse_speed_vs_size_data()
    plot_real_world_data_speed_comparison()
    # compute_iou()
    synthetic_flops_comparison()
