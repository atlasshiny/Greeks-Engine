#pragma once
#include "models/IPricingModel.hpp"
#include <cmath>
#include <vector>

// GPU macro for CUDA compatibility
#ifdef __CUDACC__
    #define HOST_DEVICE __host__ __device__
#else
    #define HOST_DEVICE
#endif

// BinomialTreeModel class implementing the binomial option pricing model
// Currently, it cannot be used with CUDA due to the use of std::vector. A custom data structure will be needed for CUDA compatibility.
// Greeks will be calculated using finite differences eventually.
class BinomialTreeModel : public IPricingModel {
public:
    // Constructor
    HOST_DEVICE BinomialTreeModel(double S, double K, double T, double r, double sigma, double q, int steps, bool isAmerican = false) 
        : S(S), K(K), T(T), r(r), sigma(sigma), q(q), steps(steps), N(steps), isAmerican(isAmerican) {}

    HOST_DEVICE inline double price(int optionType) const override {
        double dt = T / N; // Change of T with respect to the number of steps
        double u = std::exp(sigma * std::sqrt(dt)); // Up factor
        double d = 1.0 / u; // Down factor
        double p = (std::exp((r - q) * dt) - d) / (u - d); // Risk-neutral probability
        double discount = std::exp(-r * dt); // Discount factor for one time step

        // 1. Initialize terminal values at time T
        std::vector<double> values(N + 1); // SWITCH DATA STRUCTURE WHEN IMPLEMENTING CUDA
        for (int j = 0; j <= N; ++j) {
            double S_T = S * std::pow(u, j) * std::pow(d, N - j);
            values[j] = (optionType == 0) ? std::max(S_T - K, 0.0) : std::max(K - S_T, 0.0);
        }

        // 2. Backward induction
        for (int i = N - 1; i >= 0; --i) {
            for (int j = 0; j <= i; ++j) {
                // Risk-neutral expectation
                values[j] = discount * (p * values[j + 1] + (1 - p) * values[j]);
                
                if (isAmerican) {
                    // For American options, check for early exercise
                    double S_curr = S * std::pow(u, j) * std::pow(d, i - j);
                    double exerciseValue = (optionType == 0) ? std::max(S_curr - K, 0.0) : std::max(K - S_curr, 0.0);
                    values[j] = std::max(values[j], exerciseValue);
                }
            }
        }
        return values[0];
    };

    HOST_DEVICE inline Greeks calculateGreeks(int optionType) const override {
        // Implement the Greeks calculation for the binomial model here using finite differences.
        return Greeks();
    };

private:
    int N;         // Number of time steps
    double S;      // Current stock price
    double K;      // Strike price
    double T;      // Time to expiration
    double r;      // Risk-free interest rate
    double sigma;  // Volatility of the underlying asset
    double q;      // Dividend yield
    int steps;     // Number of steps in the binomial tree
    bool isAmerican; // Flag to indicate if the option is American or European
};