#include <gtest/gtest.h>
#include "BSMModel.hpp"

TEST(BSMModelTest, CallPrice) {
    double S = 100.0;   // Spot price
    double K = 100.0;   // Strike price
    double T = 1.0;     // Time to maturity in years
    double r = 0.05;    // Risk-free interest rate
    double sigma = 0.2; // Volatility of the underlying asset

    BSMModel bsm(S, K, T, r, sigma);
    double callPrice = bsm.callPrice();

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
    double putPrice = bsm.putPrice();

    double expectedPutPrice = 5.5735; 

    EXPECT_NEAR(putPrice, expectedPutPrice, 1e-4);
}