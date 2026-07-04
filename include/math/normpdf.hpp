#pragma once
#include <cmath>
#include <numbers>

double normpdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2 * std::numbers::pi);
}