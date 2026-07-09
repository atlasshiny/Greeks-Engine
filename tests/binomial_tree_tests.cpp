#include <gtest/gtest.h>
#include "models/BinomialTreeModel.hpp"
#include "models/BSMModel.hpp"
#include "Greeks.hpp"
#include <cmath>

TEST(BinomialTreeModelTest, BinomialConvergenceTest) {
    double S = 100.0;        // Spot price
    double K = 100.0;        // Strike price
    double T = 1.0;          // Time to maturity in years
    double r = 0.05;         // Risk-free interest rate
    double sigma = 0.2;      // Volatility of the underlying asset
    double q = 0.0;          // Dividend yield
    int steps = 10000;       // Number of steps in the binomial tree
    bool isAmerican = false; // Flag to indicate if the option is American or European

    BinomialTreeModel binomial(S, K, T, r, sigma, q, steps, isAmerican);
    std::vector<double> buffer(steps + 1);

    double callPriceBinomial = binomial.price(0, buffer.data()); // 0 for call option
    double putPriceBinomial = binomial.price(1, buffer.data());  // 1 for put option

    BSMModel bsm(S, K, T, r, sigma);
    double callPriceBSM = bsm.price(0); // 0 for call option
    double putPriceBSM = bsm.price(1);   // 1 for put option

    EXPECT_NEAR(callPriceBinomial, callPriceBSM, 1e-2);
    EXPECT_NEAR(putPriceBinomial, putPriceBSM, 1e-2);
}