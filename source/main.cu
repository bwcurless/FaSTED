/******************************************************************************
 * File:             main.cu
 *
 * Author:           Brian Curless
 * Created:          12/04/24
 * Description:      Responsible for parsing command line arguments, delegating to the correct
 *datasetLoader, and finding the pairs.
 *****************************************************************************/

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <math.h>
#include <vector_types.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "DataLoader/ExponentialPointGenerator.hpp"
#include "DataLoader/FilePointGenerator.hpp"
#include "DataLoader/PointListBuilder.hpp"
#include "findPairs.cuh"
#include "ptxMma.cuh"
#include "sumSquared.cuh"
#include "utils.cuh"

using SharedSize = WarpMma::SharedSize;
using InPrec = Mma::InPrec;

// Allocate the point list here so it can be reused across invocations.
Points::PointList<half_float::half> pointList;

extern "C" {
/** Run the same routine again, but with a different search radius. Useful for running quick tests
 * while fine tuning epsilon.
 *
 * \param epsilon The search radius.
 * \param savePairs Save the pairs to their own output stream.
 *
 * \returns The search results
 */
SimSearch::Results reRun(double epsilon, bool savePairs = false);

/** Load a dataset from file and compute the results.
 *
 * \param filename The name of the file that contains the dataset.
 * \param epsilon The epsilon to use.
 * \param savePairs Save the pairs to their own output stream.
 *
 * \returns The search results
 */
SimSearch::Results runFromFile(const char* filename, double epsilon, bool savePairs = false);

/** Run pair finding routine from a generated exponentially distributed dataset.
 * \param size The number of points in the dataset.
 * \param dimensionality The dimensionality of each point.
 * \param lambda The lambda of the dataset.
 * \param mean The mean value of the dataset.
 * \param epsilon The search radius.
 * \param savePairs Save the pairs to their own output stream.
 *
 * \returns The search results
 *
 */
SimSearch::Results runFromExponentialDataset(int size, int dimensionality, double lambda,
                                             double mean, double epsilon, bool savePairs = false);

/** Releases resources allocated on GPU. Needs to be called after running from file or from an
 * exponential dataset to free allocated memory. Memory persists so we can call the "reRun"
 * function.
 */
void releaseResources();
}

/** Create a set of points with monatomically increasing values. Increments by 1 for every
 * point. Note that half values can only count up to about 64000, so the max value is capped at
 * 32768.
 *
 * \param values The vector to push the values onto.
 * \param numPoints How many points to create.
 * \param numDimensions How many dimensions per point.
 *
 */
void GenerateIncreasingPoints(std::vector<half2>& values, int numPoints, int numDimensions) {
    // Kind of a hack but we go to NaN if we let it keep incrementing
    int maxFloat = 32768;
    // Fill the vector with increasing half-precision values
    // Note that this gets funny > 2048 because of imprecision of half values
    for (int m = 0; m < numPoints; m++) {
        for (int k = 0; k < numDimensions; k += 2) {
            half2 val{};
            val.x = static_cast<half>(min(maxFloat, m * numDimensions + k));
            val.y = static_cast<half>(min(maxFloat, m * numDimensions + k + 1));
            values.push_back(val);
        }
    }

    if (Debug) {
        PrintMatrix<half>("Global A", reinterpret_cast<half*>(values.data()), numPoints,
                          numDimensions);
    }
}

/** Create a set of points that is similar to an identity matrix. Will be all 0's except for
 * diagonal values.
 *
 * \param values The vector to push values onto.
 * \param numPoints How many points to create.
 * \param numDimensions How many dimensions per point.
 *
 */
void GenerateIdentityMatrixPoints(std::vector<half2>& values, int numPoints, int numDimensions) {
    // Create identity matrix
    for (int row = 0; row < numPoints; row++) {
        for (int col = 0; col < numDimensions; col += 2) {
            half2 val{0, 0};
            if (col == row)
                val.x = 1;
            else if (col + 1 == row)
                val.y = 1;
            values.push_back(val);
        }
    }

    if (Debug) {
        PrintMatrix<half>("Global B", reinterpret_cast<half*>(values.data()), numPoints,
                          numDimensions);
    }
}

double parseDouble(std::string str) {
    try {
        return std::stod(str);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument("Invalid argument: unable to parse double");
    } catch (const std::out_of_range& e) {
        throw std::out_of_range("Out of range: the number is too large or too small for a double");
    }
}

int main(int argc, char* argv[]) {
    bool foundOptionE = false;  // Flag to track if "-e" is found

    // Loop through all command-line arguments
    for (int i = 1; i < argc; ++i) {  // Start at 1 to skip the program name
        if (strcmp(argv[i], "-e") == 0) {
            foundOptionE = true;
            break;  // No need to check further, we found "-e"
        }
    }

    if (foundOptionE) {
        if (argc != 5) {
            std::cerr << "Usage: " << argv[0] << " -e <numPoints> <numDimensions> <epsilon>"
                      << std::endl;
            return 1;  // Exit with error if the number of arguments is incorrect
        }
        // Run exponential dataset
        std::cout << "Running from a generated exponential dataset";

        std::string numPointsString = argv[2];
        std::string numDimensionsString = argv[3];
        std::string epsilonString = argv[4];  // Get epsilon from the command-line argument
        int numPoints = std::stoi(numPointsString);
        int numDimensions = std::stoi(numDimensionsString);
        double epsilon = parseDouble(epsilonString);

        auto results = runFromExponentialDataset(numPoints, numDimensions, 40, 10.0, epsilon);
    } else {
        if (argc != 3) {
            std::cerr << "Usage: " << argv[0] << " <filename> <epsilon>" << std::endl;
            return 1;  // Exit with error if the number of arguments is incorrect
        }

        // Set the global locale to the default locale (which should use commas for thousands
        // separator in the US) std::cout.imbue(std::locale("en_US.UTF-8"));
        // // Set std::cout to use the "C" locale (which does not include thousands separators)
        //     std::cout.imbue(std::locale("C"));

        const char* filename = argv[1];       // Get the filename from the command-line argument
        std::string epsilonString = argv[2];  // Get epsilon from the command-line argument
        double epsilon = parseDouble(epsilonString);

        runFromFile(filename, epsilon);
    }
}

/** Runs the pair finding routine on a dataset that is passed in. The dataset could come from
 * anywhere.
 *
 * \param builder The PointListBuilder that will return the dataset.
 * \param epsilon The search radius.
 * \param savePairs Don't save the pairs off to disk.
 * \param skipPointsGeneration Don't generate new points. Use the ones from the last iteration.
 *
 * \returns The search results
 */
SimSearch::Results run(Points::PointListBuilder<half_float::half> builder, double epsilon,
                       bool savePairs, bool skipPointsGeneration) {
    // TODO this is kind of messy. Figure out a cleaner way to maintain the history here. This
    // filename isn't even correct either.
    std::string datasetName;
    // Reuse the points previously created if told to do so.
    if (!skipPointsGeneration) {
        datasetName = builder.getDatasetName();
        Mma::mmaShape bDims = SimSearch::GetBlockTileDims();
        if (Debug) {
            pointList = builder.withMaxPoints(128).build(bDims.k, bDims.m);
        } else {
            pointList = builder.build(bDims.k, bDims.m);
        }
    } else {
        datasetName = "";
    }

    // Output filename generation
    std::string outputPath =
        "/scratch/bc2497/pairsData/" + datasetName + "_" + std::to_string(epsilon);

    Mma::mmaShape paddedSearchShape{pointList.getNumPoints(), pointList.getNumPoints(),
                                    pointList.getDimensions()};
    Mma::mmaShape inputSearchShape{pointList.getActualNumPoints(), pointList.getActualNumPoints(),
                                   pointList.getActualDimensions()};

    std::cout << "Padded Search Dimensions:" << std::endl;
    std::cout << "M: " << paddedSearchShape.m << std::endl;
    std::cout << "N: " << paddedSearchShape.n << std::endl;
    std::cout << "K: " << paddedSearchShape.k << std::endl;

    std::cout << "Input Search Dimensions:" << std::endl;
    std::cout << "M: " << inputSearchShape.m << std::endl;
    std::cout << "N: " << inputSearchShape.n << std::endl;
    std::cout << "K: " << inputSearchShape.k << std::endl;

    if (Debug) {
        PrintMatrix<half>("Dataset A", reinterpret_cast<half*>(pointList.values.data()),
                          paddedSearchShape.m, paddedSearchShape.k);
    }

    auto hostParams =
        SimSearch::FindPairsParamsHost{epsilon,   paddedSearchShape,    inputSearchShape, pointList,
                                       savePairs, skipPointsGeneration, outputPath};
    SimSearch::Results results = SimSearch::FindPairs(hostParams);

    return results;
}

extern "C" {
SimSearch::Results runFromExponentialDataset(int size, int dimensionality, double lambda,
                                             double max, double epsilon, bool savePairs) {
    // Dynamically generate the dataset
    SimSearch::ExponentialPointGenerator pointGen(size, dimensionality, lambda, max);
    Points::PointListBuilder<half_float::half> pointListBuilder(&pointGen);

    // Run routine
    return run(pointListBuilder, epsilon, savePairs, false);
}

SimSearch::Results runFromFile(const char* filename, double epsilon, bool savePairs) {
    // Read dataset from file
    // Convert const char * to std::string. Ctypes can only use const char *
    std::string stringFilename(filename);
    SimSearch::FilePointGenerator pointGen(filename, ',');
    Points::PointListBuilder<half_float::half> pointListBuilder(&pointGen);

    return run(pointListBuilder, epsilon, savePairs, false);
}

SimSearch::Results reRun(double epsilon, bool savePairs) {
    // TODO don't make a dummy generator here, find a better way to bypass this.
    return run(nullptr, epsilon, savePairs, true);
}

void releaseResources() { releaseResources(); }
}
