#pragma once
#include "Greeks.hpp"
#include "Option.hpp"
#include "math/normcdf.hpp"
#include "math/normpdf.hpp"
#include "models/IPricingModel.hpp"
#include "macros.hpp"
#include <cmath>

class BSMModel : public IPricingModel {
public:
    // Constructor
    HOST_DEVICE BSMModel(double S, double K, double T, double r, double sigma) 
        : S(S), K(K), T(T), r(r), sigma(sigma) {};

    HOST_DEVICE inline double price(int optionType) const override {
        if (optionType == 0) {
            return callPrice();
        } else {
            return putPrice();
        }
    };

    HOST_DEVICE inline Greeks calculateGreeks(int optionType) const override {
        if (optionType == 0) {
            return callGreeks();
        } else {
            return putGreeks();
        }
    };

private:
    double S; // Spot price
    double K; // Strike price
    double T; // Time to maturity in years
    double r; // Risk-free interest rate
    double sigma; // Volatility of the underlying asset (annualized)

    // Helper method to calculate d1
    HOST_DEVICE inline double d1() const {
        double numerator = std::log(S / K) + (r + 0.5 * sigma * sigma) * T;
        double denominator = sigma * std::sqrt(T);
        return numerator / denominator;
    };

    // Helper method to calculate d2
    HOST_DEVICE inline double d2(double d1_val) const {
        return d1_val - sigma * std::sqrt(T);
    }

    // Helper method to calculate greeks for both call and put options (delta, gamma, vega, theta, rho)
    // This helper specifically calculates the common Greeks for both call and put options (delta, gamma, vega, rho)
    HOST_DEVICE inline void calculateCommonGreeks(bool isCall, double d1_val, double d2_val, double pdf_d1, Greeks& greeks) const {
        // Delta
        if (isCall) {
            greeks.delta = MathUtils::normcdf(d1_val);
        } else {
            greeks.delta = MathUtils::normcdf(d1_val) - 1;
        }

        // Gamma
        greeks.gamma = pdf_d1 / (S * sigma * std::sqrt(T));

        // Vega
        greeks.vega = S * pdf_d1 * std::sqrt(T);

        // Rho
        if (isCall) {
            greeks.rho = K * T * std::exp(-r * T) * MathUtils::normcdf(d2_val);
        } else {
            greeks.rho = -K * T * std::exp(-r * T) * MathUtils::normcdf(-d2_val);
        }
    }
    
    // This helper specifically calculates the theta for both call and put options
    HOST_DEVICE inline void calculateCallTheta(double d1_val, double d2_val, double pdf_d1, Greeks& greeks) const {
        greeks.theta = (-S * pdf_d1 * sigma) / (2 * std::sqrt(T)) - r * K * std::exp(-r * T) * MathUtils::normcdf(d2_val);
    }

    HOST_DEVICE inline void calculatePutTheta(double d1_val, double d2_val, double pdf_d1, Greeks& greeks) const {
        greeks.theta = (-S * pdf_d1 * sigma) / (2 * std::sqrt(T)) + r * K * std::exp(-r * T) * MathUtils::normcdf(-d2_val);
    }

    // Method to calculate the price of a call option
    HOST_DEVICE inline double callPrice() const {
        double d1_val = d1();
        double d2_val = d2(d1_val);

        return S * MathUtils::normcdf(d1_val) - K * std::exp(-r * T) * MathUtils::normcdf(d2_val);
    }

    // Method to calculate the price of a put option
    HOST_DEVICE inline double putPrice() const {
        double d1_val = d1();
        double d2_val = d2(d1_val);

        return K * std::exp(-r * T) * MathUtils::normcdf(-d2_val) - S * MathUtils::normcdf(-d1_val);
    }

    // Method to calculate the Greeks for a call option
    HOST_DEVICE inline Greeks callGreeks() const {
        Greeks greeks;

        double d1_val = d1();
        double d2_val = d2(d1_val);
        double pdf_d1 = MathUtils::normpdf(d1_val);

        calculateCommonGreeks(true, d1_val, d2_val, pdf_d1, greeks);
        calculateCallTheta(d1_val, d2_val, pdf_d1, greeks);

        return greeks;
    }

    // Method to calculate the Greeks for a put option
    HOST_DEVICE inline Greeks putGreeks() const {
        Greeks greeks;

        double d1_val = d1();
        double d2_val = d2(d1_val);
        double pdf_d1 = MathUtils::normpdf(d1_val);

        calculateCommonGreeks(false, d1_val, d2_val, pdf_d1, greeks);
        calculatePutTheta(d1_val, d2_val, pdf_d1, greeks);

        return greeks;
    }

};