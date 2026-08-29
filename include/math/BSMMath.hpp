#pragma once
#include <cmath>
#include "math/normcdf.hpp"
#include "math/normpdf.hpp"
#include "macros.hpp"

namespace BSMMath {

// Core intermediate variables
HOST_DEVICE inline double d1(double S, double K, double T, double r, double sigma) {
    double numerator = std::log(S / K) + (r + 0.5 * sigma * sigma) * T;
    double denominator = sigma * std::sqrt(T);
    return numerator / denominator;
}

HOST_DEVICE inline double d2(double d1_val, double sigma, double T) {
    return d1_val - sigma * std::sqrt(T);
}

// Pricing formulas
HOST_DEVICE inline double callPrice(double S, double K, double T, double r, double d1_val, double d2_val) {
    return S * MathUtils::normcdf(d1_val) - K * std::exp(-r * T) * MathUtils::normcdf(d2_val);
}

HOST_DEVICE inline double putPrice(double S, double K, double T, double r, double d1_val, double d2_val) {
    return K * std::exp(-r * T) * MathUtils::normcdf(-d2_val) - S * MathUtils::normcdf(-d1_val);
}

// Delta
HOST_DEVICE inline double callDelta(double d1_val) {
    return MathUtils::normcdf(d1_val);
}

HOST_DEVICE inline double putDelta(double d1_val) {
    return MathUtils::normcdf(d1_val) - 1.0;
}

// Gamma & Vega (Identical for Call and Put)
HOST_DEVICE inline double gamma(double S, double sigma, double T, double pdf_d1) {
    return pdf_d1 / (S * sigma * std::sqrt(T));
}

HOST_DEVICE inline double vega(double S, double T, double pdf_d1) {
    return S * pdf_d1 * std::sqrt(T);
}

// Theta
HOST_DEVICE inline double callTheta(double S, double K, double T, double r, double sigma, double d2_val, double pdf_d1) {
    return (-S * pdf_d1 * sigma) / (2.0 * std::sqrt(T)) - r * K * std::exp(-r * T) * MathUtils::normcdf(d2_val);
}

HOST_DEVICE inline double putTheta(double S, double K, double T, double r, double sigma, double d2_val, double pdf_d1) {
    return (-S * pdf_d1 * sigma) / (2.0 * std::sqrt(T)) + r * K * std::exp(-r * T) * MathUtils::normcdf(-d2_val);
}

// Rho
HOST_DEVICE inline double callRho(double K, double T, double r, double d2_val) {
    return K * T * std::exp(-r * T) * MathUtils::normcdf(d2_val);
}

HOST_DEVICE inline double putRho(double K, double T, double r, double d2_val) {
    return -K * T * std::exp(-r * T) * MathUtils::normcdf(-d2_val);
}

}