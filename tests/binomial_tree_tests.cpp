#include <gtest/gtest.h>
#include "models/BinomialTreeModel.hpp"
#include "models/BSMModel.hpp"
#include "Greeks.hpp"
#include <cmath>

TEST(BinomialTreeModelTest, BinomialPriceConvergenceTest) {
    double S = 100.0;        // Spot price
    double K = 100.0;        // Strike price
    double T = 1.0;          // Time to maturity in years
    double r = 0.05;         // Risk-free interest rate
    double sigma = 0.2;      // Volatility of the underlying asset
    double q = 0.0;          // Dividend yield
    int steps = 1000;       // Number of steps in the binomial tree
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

TEST(BinomialTreeModelTest, BinomialGreeksConvergenceTest) {
    double S = 100.0;        // Spot price
    double K = 100.0;        // Strike price
    double T = 1.0;          // Time to maturity in years
    double r = 0.05;         // Risk-free interest rate
    double sigma = 0.2;      // Volatility of the underlying asset
    double q = 0.0;          // Dividend yield
    int steps = 1000;       // Number of steps in the binomial tree
    bool isAmerican = false; // Flag to indicate if the option is American or European

    BinomialTreeModel binomial(S, K, T, r, sigma, q, steps, isAmerican);
    std::vector<double> buffer(steps + 1);
    double h = 0.001; // Small perturbation for finite difference

    Greeks greeksBinomialCall = binomial.calculateGreeks(0, buffer.data(), h); // 0 for call option
    Greeks greeksBinomialPut = binomial.calculateGreeks(1, buffer.data(), h);  // 1 for put option

    BSMModel bsm(S, K, T, r, sigma);
    Greeks greeksBSMCall = bsm.calculateGreeks(0); // 0 for call option
    Greeks greeksBSMPut = bsm.calculateGreeks(1);   // 1 for put option

    EXPECT_NEAR(greeksBinomialCall.delta, greeksBSMCall.delta, 1e-2);
    EXPECT_NEAR(greeksBinomialCall.gamma, greeksBSMCall.gamma, 1e-2);
    EXPECT_NEAR(greeksBinomialCall.vega, greeksBSMCall.vega, 1e-2);
    EXPECT_NEAR(greeksBinomialCall.theta, greeksBSMCall.theta, 1e-2);
    EXPECT_NEAR(greeksBinomialCall.rho, greeksBSMCall.rho, 1e-2);

    EXPECT_NEAR(greeksBinomialPut.delta, greeksBSMPut.delta, 1e-2);
    EXPECT_NEAR(greeksBinomialPut.gamma, greeksBSMPut.gamma, 1e-2);
    EXPECT_NEAR(greeksBinomialPut.vega, greeksBSMPut.vega, 1e-2);
    EXPECT_NEAR(greeksBinomialPut.theta, greeksBSMPut.theta, 1e-2);
    EXPECT_NEAR(greeksBinomialPut.rho, greeksBSMPut.rho, 1e-2);
}