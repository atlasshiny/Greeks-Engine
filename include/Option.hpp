#pragma once

struct Option {
    double K; // Strike price
    double T; // Time to maturity in years

    int type; // Option type: 0 for Call, 1 for Put
    bool isAmerican; // Whether or not a given option is an American option (used for Binomial Tree pricing)
};