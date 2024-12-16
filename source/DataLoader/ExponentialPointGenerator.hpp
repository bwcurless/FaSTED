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
    ExponentialPointGenerator(int numPoints, int dimensionality, double lambda = 40,
                              double maxValue = 1.0)
        : numPoints{numPoints},
          generatedPoints{0},
          numDim{dimensionality},
          lambda{lambda},
          maxValue{maxValue},
          gen(SEED),
          dis(lambda),
          total{0} {}

    ~ExponentialPointGenerator() {}

    std::optional<std::vector<double> > next() override {
        // Start generating random points!!!
        std::vector<double> points;
        if (generatedPoints >= numPoints) {
            return std::nullopt;
        }

        for (int j = 0; j < numDim; j++) {
            double val = 0;
            // generate value until its in the range specified
            do {
                val = dis(gen);
            } while (val < 0.0 || val > maxValue);

            total += val;

            points.push_back(val);
        }

        generatedPoints++;

        return points;
    }

    std::string getName() override {
        return "Exponential_" + std::to_string(numPoints) + "_" + std::to_string(numDim) + "_" +
               std::to_string(lambda);
    }

   private:
    int numPoints;
    int generatedPoints;
    int numDim;
    double lambda;
    double maxValue;
    std::mt19937 gen;
    std::exponential_distribution<double> dis;
    double total;
};

#endif /* end of include guard: EXPONENTIALPOINTGENERATOR_HPP_GM3KNQBC */
}
