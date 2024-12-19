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

extern "C" {
/** Load a dataset from file and compute the results.
 *
 * \param filename The name of the file that contains the dataset.
 * \param epsilon The epsilon to use.
 *
 */
SimSearch::Results runFromFile(std::string filename, double epsilon, bool skipPairs = false);

/** Run pair finding routine from a generated exponentially distributed dataset.
 * \param size The number of points in the dataset.
 * \param dimensionality The dimensionality of each point.
 * \param lambda The lambda of the dataset.
 * \param mean The mean value of the dataset.
 * \param epsilon The search radius.
 *
 * \returns The search results
 *
 */
SimSearch::Results runFromExponentialDataset(int size, int dimensionality, double lambda,
                                             double mean, double epsilon, bool skipPairs = false);
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

        std::string filename = argv[1];       // Get the filename from the command-line argument
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
 * \param skipPairs Don't save the pairs off to disk.
 *
 * \returns The search results
 */
SimSearch::Results run(Points::PointListBuilder<half_float::half> builder, double epsilon,
                       bool skipPairs) {
    // Output filename generation
    std::string outputPath = "/scratch/bc2497/pairsData/" + builder.getDatasetName() + "_" +
                             std::to_string(epsilon) + ".pairs";
    std::ofstream outFile(outputPath);
    std::cout << "Output file is: " << outputPath << std::endl;

    // Attempt to build the PointList using the provided filename
    Points::PointList<half_float::half> pointList;

    Mma::mmaShape bDims = SimSearch::GetBlockTileDims();
    if (Debug) {
        pointList = builder.withMaxPoints(128).build(bDims.k, bDims.m);
    } else {
        pointList = builder.build(bDims.k, bDims.m);
    }

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

    auto hostParams = SimSearch::FindPairsParamsHost{epsilon,   paddedSearchShape, inputSearchShape,
                                                     pointList, skipPairs,         outFile};
    SimSearch::Results results = SimSearch::FindPairs(hostParams);

    return results;
}

extern "C" {
SimSearch::Results runFromExponentialDataset(int size, int dimensionality, double lambda,
                                             double max, double epsilon, bool skipPairs) {
    // Dynamically generate the dataset
    SimSearch::ExponentialPointGenerator pointGen(size, dimensionality, lambda, max);
    Points::PointListBuilder<half_float::half> pointListBuilder(&pointGen);

    // Run routine
    return run(pointListBuilder, epsilon, skipPairs);
}

SimSearch::Results runFromFile(std::string filename, double epsilon, bool skipPairs) {
    // Read dataset from file
    SimSearch::FilePointGenerator pointGen(filename, ',');
    Points::PointListBuilder<half_float::half> pointListBuilder(&pointGen);

    return run(pointListBuilder, epsilon, skipPairs);
}
}
