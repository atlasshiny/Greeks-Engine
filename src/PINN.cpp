#include <vector>
#include <random>

#include "models/PhysicsInformedNN.hpp"
#include "models/BSMModel.hpp"
#include "Greeks.hpp"
#include "MarketParameters.hpp"
#include "Option.hpp"

// Create test cases for the Physics-Informed Neural Network (PINN) model
void createTestCases(std::vector<MarketParams>& market_params, std::vector<Option>& options, int num_cases) {
    // RNG for generating random market parameters and option data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    // Clear existing data for repeated runs
    market_params.clear();
    options.clear();

    // Reserve space for the specified number of test cases for efficiency
    market_params.reserve(num_cases);
    options.reserve(num_cases);

    // Generate random market parameters and option data  
    // Use the calculated Greeks as the expected results for the PINN model
    for (int i = 0; i < num_cases; i++) {
        MarketParams params(dis(gen), dis(gen), dis(gen), dis(gen));
        Option option(dis(gen), dis(gen), dis(gen), dis(gen));
        market_params.push_back(params);
        options.push_back(option);

        BSMModel model(market_params[i].S, options[i].K, options[i].T, market_params[i].r, market_params[i].sigma);
        Greeks greeks = model.calculateGreeks(options[i].type);
    }

    // Since vectors are passed by reference, they will contain the generated test cases when the function returns
}

// Obfuscate parts of the data to test model accuracy and robustness

void ofuscateData(std::vector<MarketParams>& market_params, std::vector<Option>& options) {
    // Randomly obfuscate parts of the data to test model accuracy and robustness
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 1);

    for (size_t i = 0; i < market_params.size(); i++) {
        if (dis(gen)) {
            market_params[i].S *= dis(gen) ? 1.1 : 0.9; // Slightly perturb the spot price
        }
        if (dis(gen)) {
            options[i].K *= dis(gen) ? 1.1 : 0.9; // Slightly perturb the strike price
        }
    }
}

// Run model evaluation on obfuscated data and compare with expected results
int main() {
    // Create vectors to hold market parameters and options for testing
    std::vector<MarketParams> market_params;
    std::vector<Option> options;

    // Generate test cases for the PINN model
    int num_cases = 1000; // Number of test cases to generate
    createTestCases(market_params, options, num_cases);

    // Obfuscate the generated data to test model accuracy and robustness
    ofuscateData(market_params, options);

    // Initialize the Physics-Informed Neural Network (PINN) model
    PhysicsInformedNN pinn_model;

    // Load pre-trained weights for the PINN model if available
    // pinn_model.load_state_dict(torch::load("pinn_model_weights.pt"));

    // Evaluate the PINN model on the obfuscated data and compare with expected results
    // This part would involve running the forward pass of the model and comparing outputs with expected Greeks

    return 0;
}

