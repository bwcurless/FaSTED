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

    with open("../ExpoDataSpeedVsSize.json", "r", encoding="utf-8") as f:
        speed_size_results = json.load(f)
    df = pd.json_normalize(speed_size_results)
    # Average results from each iteration. We used a fixed selectity here, so groupBy input shape.
    averaged_results = (
        df.groupby([INPUT_SIZE_COL, INPUT_DIM_COL], as_index=False)
        .mean()
        .drop(columns=["iteration"])
    )
    print(averaged_results)

    # Convert results to 2D array to show as a heatmap
    unique_size = np.unique(averaged_results[INPUT_SIZE_COL])
    unique_dim = np.unique(averaged_results[INPUT_DIM_COL])

    grid = np.zeros((len(unique_dim), len(unique_size)))

    for dim, size, speed in zip(
        averaged_results[INPUT_DIM_COL],
        averaged_results[INPUT_SIZE_COL],
        averaged_results[SPEED_COL],
    ):
        dim_index = np.where(unique_dim == dim)[0]
        size_index = np.where(unique_size == size)[0]
        grid[dim_index, size_index] = speed

    # TODO Fix the size ticks since I did an exponential distribution.
    fig, ax = plt.subplots()
    cax = ax.imshow(
        grid.T,
        cmap="viridis",
        origin="lower",
        aspect="auto",
        extent=(
            unique_dim[0],
            unique_dim[-1],
            unique_size[0],
            unique_size[-1],
        ),
    )
    colorbar = fig.colorbar(cax, ax=ax)
    colorbar.set_label("Speed (TFLOPS)")
    ax.set_xlabel("Dataset Dimensionality")
    ax.set_ylabel("Dataset Size (Points)")
    ax.set_title("Speed vs Input Size")
    plt.savefig("ExpoDataSpeedVsSize.pdf")
    plt.show()


if __name__ == "__main__":
    parse_selectivity_vs_speed_data()
    parse_speed_vs_size_data()
