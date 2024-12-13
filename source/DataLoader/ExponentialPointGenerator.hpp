/******************************************************************************
 * File:             ExponentialPointGenerator.hpp
 *
 * Author:           Brian Curless
 * Created:          12/12/24
 *                   Generates exponentially distributed points.
 *****************************************************************************/
#ifndef EXPONENTIALPOINTGENERATOR_HPP_GM3KNQBC
#define EXPONENTIALPOINTGENERATOR_HPP_GM3KNQBC

#define SEED 2137834274

// static seed so we can reproduce the data on other machines
#include <optional>
#include <random>

#include "PointGenerator.hpp"
namespace SimSearch {

class ExponentialPointGenerator : public SimSearch::PointGenerator {
   public:
    ExponentialPointGenerator(int numPoints, int dimensionality, double mean = 0,
                              double lambda = 40)
        : numPoints{numPoints},
          generatedPoints{0},
          numDim{dimensionality},
          mean{mean},
          lambda{lambda},
          gen(SEED),
          dis(lambda),
          total{0} {}
    ~ExponentialPointGenerator() {}

    std::optional<std::vector<double> > next() override {
        // Start generating random points!!!
        std::vector<double> point;
        if (generatedPoints >= numPoints) {
            return std::nullopt;
        }

        for (int j = 0; j < numDim; j++) {
            double val = 0;
            // generate value until its in the range 0-1
            do {
                val = dis(gen);
            } while (val < 0 || val > 1);

            total += val;

            point.push_back(val);
        }

        generatedPoints++;

        return point;
    }

    std::string getName() override {
        return "Exponential_" + std::to_string(numPoints) + "_" + std::to_string(numDim) + "_" +
               std::to_string(lambda);
    }

   private:
    int numPoints;
    int generatedPoints;
    int numDim;
    double mean;
    double lambda;
    std::mt19937 gen;
    std::exponential_distribution<double> dis;
    double total;
};

#endif /* end of include guard: EXPONENTIALPOINTGENERATOR_HPP_GM3KNQBC */
}
