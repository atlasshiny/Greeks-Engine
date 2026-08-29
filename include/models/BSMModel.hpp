#pragma once
#include "Greeks.hpp"
#include "Option.hpp"
#include "macros.hpp"
#include "math/normpdf.hpp"
#include "math/BSMMath.hpp"

class BSMModel {
public:
    // Constructor
    HOST_DEVICE BSMModel(double S, double K, double T, double r, double sigma) 
        : S(S), K(K), T(T), r(r), sigma(sigma) {};

    HOST_DEVICE inline double price(int optionType) const {
        return (optionType == 0) ? callPrice() : putPrice();
    };

    HOST_DEVICE inline Greeks calculateGreeks(int optionType) const {
        return (optionType == 0) ? callGreeks() : putGreeks();
    };

    // Used to update the model parameters for batch processing without creating a new instance
    HOST_DEVICE inline void setParameters(double S_, double K_, double T_, double r_, double sigma_) {
        S = S_;
        K = K_;
        T = T_;
        r = r_;
        sigma = sigma_;
    }

private:
    double S; // Spot price
    double K; // Strike price
    double T; // Time to maturity in years
    double r; // Risk-free interest rate
    double sigma; // Volatility

    // Method to calculate the price of a call option
    HOST_DEVICE inline double callPrice() const {
        double d1_val = BSMMath::d1(S, K, T, r, sigma);
        double d2_val = BSMMath::d2(d1_val, sigma, T);
        return BSMMath::callPrice(S, K, T, r, d1_val, d2_val);
    }

    // Method to calculate the price of a put option
    HOST_DEVICE inline double putPrice() const {
        double d1_val = BSMMath::d1(S, K, T, r, sigma);
        double d2_val = BSMMath::d2(d1_val, sigma, T);
        return BSMMath::putPrice(S, K, T, r, d1_val, d2_val);
    }

    // Method to calculate the Greeks for a call option
    HOST_DEVICE inline Greeks callGreeks() const {
        Greeks greeks;
        double d1_val = BSMMath::d1(S, K, T, r, sigma);
        double d2_val = BSMMath::d2(d1_val, sigma, T);
        double pdf_d1 = MathUtils::normpdf(d1_val);

        greeks.delta = BSMMath::callDelta(d1_val);
        greeks.gamma = BSMMath::gamma(S, sigma, T, pdf_d1);
        greeks.vega  = BSMMath::vega(S, T, pdf_d1);
        greeks.theta = BSMMath::callTheta(S, K, T, r, sigma, d2_val, pdf_d1);
        greeks.rho   = BSMMath::callRho(K, T, r, d2_val);

        return greeks;
    }

    // Method to calculate the Greeks for a put option
    HOST_DEVICE inline Greeks putGreeks() const {
        Greeks greeks;
        double d1_val = BSMMath::d1(S, K, T, r, sigma);
        double d2_val = BSMMath::d2(d1_val, sigma, T);
        double pdf_d1 = MathUtils::normpdf(d1_val);

        greeks.delta = BSMMath::putDelta(d1_val);
        greeks.gamma = BSMMath::gamma(S, sigma, T, pdf_d1);
        greeks.vega  = BSMMath::vega(S, T, pdf_d1);
        greeks.theta = BSMMath::putTheta(S, K, T, r, sigma, d2_val, pdf_d1);
        greeks.rho   = BSMMath::putRho(K, T, r, d2_val);

        return greeks;
    }

};

struct BSMPolicy {
    MODEL_POLICY float operator()(float S, float K, float T, float r, float sigma, int optionType) const {
        BSMModel model(S, K, T, r, sigma);
        return static_cast<float>(model.price(optionType));
    }
};