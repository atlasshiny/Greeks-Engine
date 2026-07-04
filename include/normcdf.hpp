#pragma once
#include <cmath>

double normcdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}