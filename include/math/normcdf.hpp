#pragma once
#include <cmath>
#include "macros.hpp"

namespace MathUtils {
    HOST_DEVICE inline double normcdf(double x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    }
}