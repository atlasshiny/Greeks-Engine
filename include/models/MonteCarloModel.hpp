#pragma once
#include "macros.hpp"
#include "math/gbm.hpp"

class MonteCarloModel {
public:
    HOST_DEVICE MonteCarloModel(double S, double K, double T, double r, double sigma, int numSimulations)
        : S(S), K(K), T(T), r(r), sigma(sigma), numSimulations(numSimulations) {}

    #ifdef __CUDACC__
        // Device-side price call for CUDA threads
        DEVICE inline double price(int optionType, curandState_t state) const {
            double totalPayoff = 0.0;
            for (long i = 0; i < numSimulations; ++i) {
                double ST = gbm.sampleTerminal(T, &state);
                double payoff = (optionType == 0) ? fmax(ST - K, 0.0) : fmax(K - ST, 0.0);
                totalPayoff += payoff;
            }
            return std::exp(-r * T) * (totalPayoff / numSimulations);
        }
    #else
        // Host-side price call for CPU
        template <typename RNG>
        inline double price(int optionType, RNG& gen, std::normal_distribution<double>& dist) const {
            double totalPayoff = 0.0;
            for (long i = 0; i < numSimulations; ++i) {
                double ST = gbm.sampleTerminal(T, gen, dist);
                double payoff = (optionType == 0) ? std::max(ST - K, 0.0) : std::max(K - ST, 0.0);
                totalPayoff += payoff;
            }
            return std::exp(-r * T) * (totalPayoff / numSimulations);
        }
    #endif

    // Used to update the model parameters for batch processing without creating a new instance
    HOST_DEVICE inline void setParameters(double S_, double K_, double T_, double r_, double sigma_, int numSimulations_) {
        S = S_;
        K = K_;
        T = T_;
        r = r_;
        sigma = sigma_;
        numSimulations = numSimulations_;
        gbm = MathUtils::GBMProcess(S, r, sigma);
    }

    // Note the lack of a calculateGreeks method. If they were to be calculated, they would likely be calculated using 
    // finite differences similar to the BinomialTreeModel class. Because it is substantially more expensive to calculate Greeks 
    // using Monte Carlo simulations, they are not implemented here for now (if they are ever implemented).

private:
    double S;      // Current stock price
    double K;      // Strike price
    double T;      // Time to expiration
    double r;      // Risk-free interest rate
    double sigma;  // Volatility of the underlying asset
    int numSimulations; // Number of Monte Carlo simulations
    MathUtils::GBMProcess gbm{S, r, sigma}; // GBM process for simulating stock prices

};