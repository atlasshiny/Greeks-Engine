#include <gtest/gtest.h>
#include "models/BSMModel.hpp"
#include "Greeks.hpp"

TEST(BSMModelTest, CallPrice) {
    double S = 100.0;   // Spot price
    double K = 100.0;   // Strike price
    double T = 1.0;     // Time to maturity in years
    double r = 0.05;    // Risk-free interest rate
    double sigma = 0.2; // Volatility of the underlying asset

    BSMModel bsm(S, K, T, r, sigma);
    double callPrice = bsm.price(0); // 0 for call option

    double expectedCallPrice = 10.4506; 

    EXPECT_NEAR(callPrice, expectedCallPrice, 1e-4);
}

TEST(BSMModelTest, PutPrice) {
    double S = 100.0;   // Spot price
    double K = 100.0;   // Strike price
    double T = 1.0;     // Time to maturity in years
    double r = 0.05;    // Risk-free interest rate
    double sigma = 0.2; // Volatility of the underlying asset

    BSMModel bsm(S, K, T, r, sigma);
    double putPrice = bsm.price(1); // 1 for put option

    double expectedPutPrice = 5.5735; 

    EXPECT_NEAR(putPrice, expectedPutPrice, 1e-4);
}

TEST(BSMModelTest, CallGreeks) {
    double S = 100.0;   // Spot price
    double K = 100.0;   // Strike price
    double T = 1.0;     // Time to maturity in years
    double r = 0.05;    // Risk-free interest rate
    double sigma = 0.2; // Volatility of the underlying asset

    BSMModel bsm(S, K, T, r, sigma);
    Greeks callGreeks = bsm.calculateGreeks(0); // 0 for call option

    EXPECT_NEAR(callGreeks.delta, 0.6368, 1e-4);
    EXPECT_NEAR(callGreeks.gamma, 0.0188, 1e-4);
    EXPECT_NEAR(callGreeks.vega, 37.5240, 1e-4);
    EXPECT_NEAR(callGreeks.theta, -6.4140, 1e-4);
    EXPECT_NEAR(callGreeks.rho, 53.2325, 1e-4);
}