#include "BSMModel.hpp"
#include "normcdf.hpp"
#include <cmath>
#include <math.h>

BSMModel::BSMModel(double S, double K, double T, double r, double sigma)
    : S(S), K(K), T(T), r(r), sigma(sigma) {}

double BSMModel::callPrice() const {
    return S * normcdf(d1()) - K * std::exp(-r * T) * normcdf(d2());
}

double BSMModel::putPrice() const {
    return K * std::exp(-r * T) * normcdf(-d2()) - S * normcdf(-d1());
}

double BSMModel::d1() const {
    double numerator = std::log(S / K) + (r + 0.5 * sigma * sigma) * T;
    double denominator = sigma * std::sqrt(T);
    return numerator / denominator;
}

double BSMModel::d2() const {
    return d1() - sigma * std::sqrt(T);
}