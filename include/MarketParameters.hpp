#pragma once

struct MarketParams {
    double S;     // Spot price
    double r;     // Risk-free interest rate
    double sigma; // Volatility of the underlying asset (annualized)
    double q;     // Dividend yield
};