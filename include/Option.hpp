#pragma once

struct Option {
    double K; // Strike price
    double T; // Time to maturity in years

    int type; // Option type: 0 for Call, 1 for Put
};