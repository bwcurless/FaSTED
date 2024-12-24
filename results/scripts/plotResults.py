import json
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Map out column names for later
selCol = "selectivity"
speedCol = "results.TFLOPS"
inputSizeCol = "results.inputProblemShape.m"
inputDimCol = "results.inputProblemShape.k"


def parseSelectivityVsSpeedData():
    with open("../selectivityVsSpeed.json", "r") as f:
        selec_speed_results = json.load(f)
    # Pretty print json
    # print(json.dumps(selec_speed_results, indent=4))

    df = pd.json_normalize(selec_speed_results)
    # Average results from each iteration
    averagedResults = (
        df.groupby(selCol, as_index=False).mean().drop(columns=["iteration"])
    )
    print(averagedResults)
    plt.figure(1)
    plt.plot(averagedResults[selCol], averagedResults[speedCol], marker=".")
    plt.xlabel("Selectivity")
    plt.ylabel("TFLOPS")
    plt.title("Selectivity vs Throughput")
    plt.savefig("selectivityVsSpeed.pdf")
    plt.show()


def parseSpeedVsSizeData():
    with open("../ExpoDataSpeedVsSize.json", "r") as f:
        speed_size_results = json.load(f)
    df = pd.json_normalize(speed_size_results)
    # Average results from each iteration. We used a fixed selectity here, so groupBy input shape.
    averagedResults = (
        df.groupby([inputSizeCol, inputDimCol], as_index=False)
        .mean()
        .drop(columns=["iteration"])
    )
    print(averagedResults)
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(
        averagedResults[inputSizeCol],
        averagedResults[inputDimCol],
        averagedResults[speedCol],
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
    parseSelectivityVsSpeedData()
    parseSpeedVsSizeData()
