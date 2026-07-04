#include "BSMModel.hpp"

BSMModel::BSMModel(double S, double K, double T, double r, double sigma)
    : S(S), K(K), T(T), r(r), sigma(sigma) {}

double BSMModel::callPrice() const {
    // Implementation for call option price
}

double BSMModel::putPrice() const {
    // Implementation for put option price
}

double BSMModel::d1() const {
    // Implementation for d1
}

double BSMModel::d2() const {
    // Implementation for d2
}