#pragma once
#include "macros.hpp"

class MonteCarloModel {
public:
    HOST_DEVICE MonteCarloModel(double S, double K, double T, double r, double sigma, double q, int numSimulations)
        : S(S), K(K), T(T), r(r), sigma(sigma), q(q), numSimulations(numSimulations) {}

    HOST_DEVICE inline double price() {
        // Placeholder for Monte Carlo pricing logic
        return 0.0;
    };

    // Note the lack of a calculateGreeks method. If they were to be calculated, they would likely be calculated using 
    // finite differences similar to the BinomialTreeModel class. Because it is substantially more expensive to calculate Greeks 
    // using Monte Carlo simulations, they are not implemented here for now (if they are ever implemented).

private:
    double S;      // Current stock price
    double K;      // Strike price
    double T;      // Time to expiration
    double r;      // Risk-free interest rate
    double sigma;  // Volatility of the underlying asset
    double q;      // Dividend yield
    int numSimulations; // Number of Monte Carlo simulations

};