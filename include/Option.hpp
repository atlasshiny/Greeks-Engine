#pragma once

struct Option {
    double S; // Spot price
    double K; // Strike price
    double T; // Time to maturity in years
    double r; // Risk-free interest rate
    double sigma; // Volatility of the underlying asset (annualized)

    int type; // Option type: 0 for Call, 1 for Put
};