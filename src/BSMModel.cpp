#include "BSMModel.hpp"
#include "normcdf.hpp"
#include "normpdf.hpp"
#include "Greeks.hpp"
#include <cmath>

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

void BSMModel::calculateCommonGreeks(bool isCall, double d1_val, double d2_val, double pdf_d1, Greeks& greeks) const {
    // Delta
    if (isCall) {
        greeks.delta = normcdf(d1_val);
    } else {
        greeks.delta = normcdf(d1_val) - 1;
    }

    // Gamma
    greeks.gamma = pdf_d1 / (S * sigma * std::sqrt(T));

    // Vega
    greeks.vega = S * pdf_d1 * std::sqrt(T);

    // Rho
    if (isCall) {
        greeks.rho = K * T * std::exp(-r * T) * normcdf(d2_val);
    } else {
        greeks.rho = -K * T * std::exp(-r * T) * normcdf(-d2_val);
    }
}

void BSMModel::calculateCallTheta(double d1_val, double d2_val, double pdf_d1, Greeks& greeks) const {
    greeks.theta = (-S * pdf_d1 * sigma) / (2 * std::sqrt(T)) - r * K * std::exp(-r * T) * normcdf(d2_val);
}

void BSMModel::calculatePutTheta(double d1_val, double d2_val, double pdf_d1, Greeks& greeks) const {
    greeks.theta = (-S * pdf_d1 * sigma) / (2 * std::sqrt(T)) + r * K * std::exp(-r * T) * normcdf(-d2_val);
}

Greeks BSMModel::callGreeks() const {
    Greeks greeks;

    double d1_val = d1();
    double d2_val = d2();
    double pdf_d1 = normpdf(d1_val);

    calculateCommonGreeks(true, d1_val, d2_val, pdf_d1, greeks);
    calculateCallTheta(d1_val, d2_val, pdf_d1, greeks);

    return greeks;
}

Greeks BSMModel::putGreeks() const {
    Greeks greeks;

    double d1_val = d1();
    double d2_val = d2();
    double pdf_d1 = normpdf(d1_val);

    calculateCommonGreeks(false, d1_val, d2_val, pdf_d1, greeks);
    calculatePutTheta(d1_val, d2_val, pdf_d1, greeks);

    return greeks;
}