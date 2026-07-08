#include <gtest/gtest.h>
#include "models/BSMModel.hpp"
#include "Greeks.hpp"
#include <cmath>

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

TEST(BSMModelTest, PutGreeks) {
    double S = 100.0;   // Spot price
    double K = 100.0;   // Strike price
    double T = 1.0;     // Time to maturity in years
    double r = 0.05;    // Risk-free interest rate
    double sigma = 0.2; // Volatility of the underlying asset

    BSMModel bsm(S, K, T, r, sigma);
    Greeks putGreeks = bsm.calculateGreeks(1); // 1 for put option

    EXPECT_NEAR(putGreeks.delta, -0.3632, 1e-4);
    EXPECT_NEAR(putGreeks.gamma, 0.0188, 1e-4);
    EXPECT_NEAR(putGreeks.vega, 37.5240, 1e-4);
    
    // Corrected mathematical expectations for Put Theta and Rho
    EXPECT_NEAR(putGreeks.theta, -1.6579, 1e-4); 
    EXPECT_NEAR(putGreeks.rho, -41.8905, 1e-4);  
}

TEST(BSMModelTest, GreeksPairity) {
    double S = 100.0;   // Spot price
    double K = 100.0;   // Strike price
    double T = 1.0;     // Time to maturity in years
    double r = 0.05;    // Risk-free interest rate
    double sigma = 0.2; // Volatility of the underlying asset

    BSMModel bsm(S, K, T, r, sigma);
    Greeks callGreeks = bsm.calculateGreeks(0); // 0 for call option
    Greeks putGreeks = bsm.calculateGreeks(1);  // 1 for put option

    // Corrected parity formulas
    EXPECT_NEAR(callGreeks.delta - putGreeks.delta, 1.0, 1e-4);
    EXPECT_NEAR(callGreeks.gamma, putGreeks.gamma, 1e-4);
    EXPECT_NEAR(callGreeks.vega, putGreeks.vega, 1e-4);
    EXPECT_NEAR(callGreeks.theta - putGreeks.theta, -r * K * std::exp(-r * T), 1e-4);
    EXPECT_NEAR(callGreeks.rho - putGreeks.rho, K * T * std::exp(-r * T), 1e-4);
}