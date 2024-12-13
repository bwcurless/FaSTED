/******************************************************************************
 * File:             PointGenerator.hpp
 *
 * Author:           Brian Curless
 * Created:          12/11/24
 * Description:      Abstraction for obtaining a sequence of points.
 *****************************************************************************/
#ifndef POINTGENERATOR_HPP_XGFNJWTL
#define POINTGENERATOR_HPP_XGFNJWTL

#include <optional>
#include <vector>

namespace SimSearch {

/** An abstract base class for a point generator.
 */
class PointGenerator {
   public:
    virtual ~PointGenerator() = default;

    /** Return the next point in the sequence. If there are no more points, returns null.
     */
    virtual std::optional<std::vector<double>> next() = 0;
};

}  // namespace SimSearch

#endif /* end of include guard: POINTGENERATOR_HPP_XGFNJWTL */
