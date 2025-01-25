"""
File: plotResults.py
Description: Plots experimental results for publishing.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Map out column names for later
SEL_COL = "selectivity"
SPEED_COL = "results.TFLOPS"
INPUT_SIZE_COL = "results.inputProblemShape.m"
INPUT_DIM_COL = "results.inputProblemShape.k"


def parse_selectivity_vs_speed_data():
    """Read in the selectivity vs speed data and create plots for it"""

    with open("../selectivityVsSpeed.json", "r", encoding="utf-8") as f:
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
    plt.xlabel("Selectivity")
    plt.ylabel("TFLOPS")
    plt.ylim(0, 300)
    plt.title("Selectivity vs Throughput")
    plt.savefig("selectivityVsSpeed.pdf")
    plt.show()


def parse_speed_vs_size_data():
    """Read in the speed vs size data and create plots for it."""

    with open(
        "../ExpoDataSpeedVsSize_OnDemandRasterizer.json", "r", encoding="utf-8"
    ) as f:
        speed_size_results = json.load(f)
    df = pd.json_normalize(speed_size_results)
    # Average results from each iteration. We used a fixed selectity here, so groupBy input shape.
    averaged_results = (
        df.groupby([INPUT_SIZE_COL, INPUT_DIM_COL], as_index=False)
        .mean()
        .drop(columns=["iteration"])
    )
    print(averaged_results)

    # Extract x and y labels...
    unique_size = np.unique(averaged_results[INPUT_SIZE_COL])
    unique_dim = np.unique(averaged_results[INPUT_DIM_COL])

    num_sizes_tested = len(unique_size)
    num_dims_tested = len(unique_dim)

    max_speed = averaged_results[SPEED_COL].max()

    # Convert results to 2D array to show as a heatmap
    grid = np.zeros((num_dims_tested, num_sizes_tested))

    for dim, size, speed in zip(
        averaged_results[INPUT_DIM_COL],
        averaged_results[INPUT_SIZE_COL],
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
        extent=(
            0,
            num_dims_tested,
            0,
            num_sizes_tested,
        ),
    )

    # To center labels on pixels, must offset by a half a pixel
    center_offset = 0.5

    ax.set_xlabel("Dataset Dimensionlity")
    ax.set_xticks(np.arange(num_dims_tested) + center_offset)
    ax.set_xticklabels(unique_dim)

    ax.set_ylabel("Dataset Size (Points)")
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
    colorbar.set_label("Speed (TFLOPS)")
    ax.set_title("Algorithm Performance")
    plt.tight_layout()
    # plt.savefig("ExpoDataSpeedVsSize.pdf")
    plt.show()


if __name__ == "__main__":
    # parse_selectivity_vs_speed_data()
    parse_speed_vs_size_data()
