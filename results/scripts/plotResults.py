"""
File: plotResults.py
Description: Plots experimental results for publishing.
"""

import json
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
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(
        averaged_results[INPUT_SIZE_COL],
        averaged_results[INPUT_DIM_COL],
        averaged_results[SPEED_COL],
        cmap="viridis",
    )
    ax.set_xlabel("Dataset Size (Points)")
    ax.set_ylabel("Dataset Dimensionality")
    ax.set_zlabel("Speed (TFLOPS)")
    ax.set_title("Speed vs Dataset Shape")
    fig.colorbar(surf)
    plt.savefig("ExpoDataSpeedVsSize.pdf")
    plt.show()


if __name__ == "__main__":
    parse_selectivity_vs_speed_data()
    parse_speed_vs_size_data()
