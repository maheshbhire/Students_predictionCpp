#ifndef MLP_HPP
#define MLP_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;

class MLP {
private:
    FFN<NegativeLogLikelihood<>, RandomInitialization> model;
    int epochs;
    double lr;
    bool trained;

public:
    MLP(int max_epochs = 100, double learning_rate = 0.01)
        : epochs(max_epochs), lr(learning_rate), trained(false) {}

    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
        if (X.empty()) return;

        int inputSize = X[0].size();
        int numClasses = *std::max_element(y.begin(), y.end()) + 1;

        // Clear previous layers if any
        model.ClearLayers();

        // Add layers
        model.Add<Linear<> >(inputSize, 10);
        model.Add<ReLULayer<> >();
        model.Add<Linear<> >(10, numClasses);
        model.Add<LogSoftMax<> >();

        // Convert data to arma matrices (features as columns)
        mat Xmat(inputSize, X.size());
        Row<size_t> ymat(X.size());

        for (size_t i = 0; i < X.size(); ++i) {
            for (size_t j = 0; j < inputSize; ++j) {
                Xmat(j, i) = X[i][j];
            }
            ymat(i) = y[i];
        }

        // Setup optimizer
        ens::StandardSGD optimizer(
            lr,    // Step size
            32,    // Batch size (mini-batch)
            epochs,// Max iterations (epochs)
            1e-5,  // Tolerance
            true   // Shuffle
        );

        // Train the model
        model.Train(Xmat, ymat, optimizer);

        trained = true;
    }

    int predict(const std::vector<double>& x) {
        if (!trained || x.empty()) return -1;

        mat xmat(x.size(), 1);
        for (size_t i = 0; i < x.size(); ++i) {
            xmat(i, 0) = x[i];
        }

        mat output;
        model.Predict(xmat, output);

        // Get the index of the max element in output (predicted class)
        uword predicted_label;
        output.col(0).max(predicted_label);

        return (int)predicted_label;
    }

    std::vector<int> predict_batch(const std::vector<std::vector<double>>& X) {
        std::vector<int> preds;
        for (const auto& x : X) {
            preds.push_back(predict(x));
        }
        return preds;
    }
};

#endif // MLP_HPP