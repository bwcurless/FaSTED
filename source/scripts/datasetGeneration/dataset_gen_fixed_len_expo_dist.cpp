// We make exponentially distributed datasets
// Needed so that we get more neighbors in comparison to uniformly distributed data
// make in the range 0-1, so we don't need to re-normalize for Super-EGO

// Makefile:
// g++ -std=c++17 -O3 dataset_gen_fixed_len_expo_dist.cpp -o dataset_gen_fixed_len_expo_dist -lstdc++fs
//

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

// static seed so we can reproduce the data on other machines
#define SEED 2137834274

using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 4) {
        cout
            << "\n\nIncorrect number of input parameters.  \nShould be: num. dimensions data "
               "points\n E.g., ./dataset_gen_fixed_len_expo_dist.cpp 2 2000000 ./outputDirectory\n";
        return 1;
    }

    char inputnumdim[500];
    char inputdatapoints[500];

    strcpy(inputnumdim, argv[1]);
    strcpy(inputdatapoints, argv[2]);
    std::filesystem::path outputDirectory(argv[3]);

    unsigned int numDim = atoi(inputnumdim);
    unsigned int dataPoints = atoi(inputdatapoints);

    unsigned int length = 1;
    // unsigned int numDim=8;
    // unsigned int dataPoints=100000000;

    printf("\nTotal datapoints: %d", dataPoints);
    double datasetsize = ((dataPoints * 8.0 * numDim) / (1024.0 * 1024.0));
    printf("Size of dataset (MiB): %f", datasetsize);

    // for my file formatting
    std::ofstream myfile;
    std::string fname = "dataset_fixed_len_pts_expo_NDIM_";
    fname += std::to_string(numDim);
    fname += "_pts_";
    fname += std::to_string(dataPoints);
    fname += ".txt";
    myfile.open(outputDirectory / fname);

    // for SuperEGO file formatting
    std::ofstream myfile2;
    std::string fname2 = "dataset_fixed_len_pts_expo_NDIM_";
    fname2 += std::to_string(numDim);
    fname2 += "_pts_";
    fname2 += std::to_string(dataPoints);
    fname2 += "_SUPEREGO.txt";
    myfile2.open(outputDirectory / fname2);

    // std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(SEED);  // Standard mersenne_twister_engine seeded with rd()
    std::exponential_distribution<double> dis(40.0);  // lambda=1.5
    double total = 0;

    for (int i = 0; i < dataPoints; i++) {
        myfile2 << i + 1 << " ";  // enumerate each line
        for (int j = 0; j < numDim; j++) {
            double val = 0;
            // generate value until its in the range 0-1
            do {
                val = dis(gen) * length;
            } while (val < 0 || val > 1);

            total += val;

            if (j < numDim - 1) {
                myfile << val << ", ";  // my formatting
                myfile2 << val << " ";  // SuperEGO formatting
            } else {
                myfile << val;   // my formatting
                myfile2 << val;  // SuperEGO formatting
            }
        }
        myfile << std::endl;
        myfile2 << std::endl;
    }

    printf("\nAverage of values generated: %f", total / (dataPoints * numDim * 1.0));

    myfile.close();
    myfile2.close();
    printf("\n\n");

    return 0;
}
