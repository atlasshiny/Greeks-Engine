#pragma once
#include "Greeks.hpp"
#include "Option.hpp"
#include <cmath>
#include <vector>
#include "macros.hpp"

// BinomialTreeModel class implementing the binomial option pricing model
// Currently, it cannot be used with CUDA due to the use of std::vector. A custom data structure will be needed for CUDA compatibility.
// Greeks will be calculated using finite differences eventually.
class BinomialTreeModel {
public:
    // Constructor
    HOST_DEVICE BinomialTreeModel(double S, double K, double T, double r, double sigma, double q, int steps, bool isAmerican = false) 
        : S(S), K(K), T(T), r(r), sigma(sigma), q(q), N(steps), isAmerican(isAmerican) {}

    // Method to calculate the price of an option using the binomial tree model
    // A buffer is passed to avoid dynamic memory allocation in CUDA. The buffer should be of size N+1.
    HOST_DEVICE inline double price(int optionType, double* buffer) const {
        return price_internal(optionType, buffer, S, sigma, r, q, T);
    };

    // Method to calculate the Greeks of an option using finite differences
    HOST_DEVICE inline Greeks calculateGreeks(int optionType, double* buffer, double h) const {
        // Delta
        double delta = calculateDelta(optionType, buffer, h);

        // Gamma
        double gamma = calculateGamma(optionType, buffer, h);

        // Vega
        double vega = calculateVega(optionType, buffer, h);

        // Theta
        double theta = calculateTheta(optionType, buffer, h);

        // Rho
        double rho = calculateRho(optionType, buffer, h);

        return Greeks{delta, gamma, vega, theta, rho};
    };

private:
    int N;         // Number of time steps
    double S;      // Current stock price
    double K;      // Strike price
    double T;      // Time to expiration
    double r;      // Risk-free interest rate
    double sigma;  // Volatility of the underlying asset
    double q;      // Dividend yield
    bool isAmerican; // Flag to indicate if the option is American or European

    // Internal method to calculate the price of an option and can also take in modified parameters for Greeks calculation.
    HOST_DEVICE inline double price_internal(int optionType, double* buffer, double S_val, double sigma_val, double r_val, double q_val, double T_val) const {
        double dt = T_val / N; // Change of T with respect to the number of steps
        double u = std::exp(sigma_val * std::sqrt(dt)); // Up factor
        double d = 1.0 / u; // Down factor
        double p = (std::exp((r_val - q_val) * dt) - d) / (u - d); // Risk-neutral probability
        double discount = std::exp(-r_val * dt); // Discount factor for one time step

        // 1. Initialize terminal values at time T
        for (int j = 0; j <= N; ++j) {
            double S_T = S_val * std::pow(u, j) * std::pow(d, N - j);
            buffer[j] = (optionType == 0) ? fmax(S_T - K, 0.0) : fmax(K - S_T, 0.0);
        }

        // 2. Backward induction
        for (int i = N - 1; i >= 0; --i) {
            for (int j = 0; j <= i; ++j) {
                // Risk-neutral expectation
                buffer[j] = discount * (p * buffer[j + 1] + (1 - p) * buffer[j]);
                
                if (isAmerican) {
                    // For American options, check for early exercise
                    double S_curr = S_val * std::pow(u, j) * std::pow(d, i - j);
                    double exerciseValue = (optionType == 0) ? fmax(S_curr - K, 0.0) : fmax(K - S_curr, 0.0);
                    buffer[j] = fmax(buffer[j], exerciseValue);
                }
            }
        }
        return buffer[0];
    }

    // Method to calculate the first order finite difference (specifically using central difference) for Greeks calculation
    template <typename Function>
    HOST_DEVICE inline double firstCentralDifference(Function f, double x, int optionType, double* buffer, double delta) const {
        double h = delta;
        return (f(x + h) - f(x - h)) / (2.0 * h);
    }

    // Method to calculate the second order finite difference (specifically using central difference) for Greeks calculation
    template <typename Function>
    HOST_DEVICE inline double secondCentralDifference(Function f, double x, int optionType, double* buffer, double delta) const {
        double h = delta;
        return (f(x + h) - (2.0 * f(x)) + f(x - h)) / (h * h);
    }

    HOST_DEVICE inline double calculateDelta(int optionType, double* buffer, double delta) const {
        // Use a lambda function to capture the current state and compute the price for different stock prices
        auto priceFunc = [this, optionType, buffer](double S_val) {
            return this->price_internal(optionType, buffer, S_val, sigma, r, q, T);
        };
        return firstCentralDifference(priceFunc, S, optionType, buffer, delta);
    }

    HOST_DEVICE inline double calculateGamma(int optionType, double* buffer, double delta) const {
        // Use a lambda function to capture the current state and compute the price for different stock prices
        auto priceFunc = [this, optionType, buffer](double S_val) {
            return this->price_internal(optionType, buffer, S_val, sigma, r, q, T);
        };
        return secondCentralDifference(priceFunc, S, optionType, buffer, delta);
    }

    HOST_DEVICE inline double calculateVega(int optionType, double* buffer, double delta) const {
        // Use a lambda function to capture the current state and compute the price for different volatilities
        auto priceFunc = [this, optionType, buffer](double sigma_val) {
            return this->price_internal(optionType, buffer, S, sigma_val, r, q, T);
        };
        return firstCentralDifference(priceFunc, sigma, optionType, buffer, delta);
    }

    HOST_DEVICE inline double calculateTheta(int optionType, double* buffer, double delta) const {
        // Use a lambda function to capture the current state and compute the price for different times to expiration
        auto priceFunc = [this, optionType, buffer](double T_val) {
            return this->price_internal(optionType, buffer, S, sigma, r, q, T_val);
        };
        return firstCentralDifference(priceFunc, T, optionType, buffer, delta);
    }

    HOST_DEVICE inline double calculateRho(int optionType, double* buffer, double delta) const {
        // Use a lambda function to capture the current state and compute the price for different interest rates
        auto priceFunc = [this, optionType, buffer](double r_val) {
            return this->price_internal(optionType, buffer, S, sigma, r_val, q, T);
        };
        return firstCentralDifference(priceFunc, r, optionType, buffer, delta);
    }

};