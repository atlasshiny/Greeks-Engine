#include <gtest/gtest.h>
#include "models/MonteCarloModel.hpp"
#include "models/BSMModel.hpp"
#include "Greeks.hpp"
#include <cmath>
#include <random>

TEST(MonteCarloModelTest, MonteCarloPriceConvergenceTest) {
    double S = 100.0;        // Spot price
    double K = 100.0;        // Strike price
    double T = 1.0;          // Time to maturity in years
    double r = 0.05;         // Risk-free interest rate
    double sigma = 0.2;      // Volatility of the underlying asset
    int numSimulations = 100000; // Number of Monte Carlo simulations

    std::random_device rd;
    std::normal_distribution<double> dist(0.0, 1.0);

    MonteCarloModel monteCarlo(S, K, T, r, sigma, numSimulations);

    double callPriceMonteCarlo = monteCarlo.price(0, rd, dist);
    double putPriceMonteCarlo = monteCarlo.price(1, rd, dist);

    BSMModel bsm(S, K, T, r, sigma);
    double callPriceBSM = bsm.price(0); // 0 for call option
    double putPriceBSM = bsm.price(1);   // 1 for put option

    EXPECT_NEAR(callPriceMonteCarlo, callPriceBSM, 1.0); // Allowing a larger tolerance due to Monte Carlo variability
    EXPECT_NEAR(putPriceMonteCarlo, putPriceBSM, 1.0); // Allowing a larger tolerance due to Monte Carlo variability

}