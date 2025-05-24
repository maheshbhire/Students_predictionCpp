#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

#include <vector>
#include <cmath>
#include <numeric>

using namespace std;

class LogisticRegression {
    vector<double> weights;
    double bias;
    double lr;
    int epochs;

public:
    LogisticRegression(double learning_rate = 0.01, int max_epochs = 1000)
        : lr(learning_rate), epochs(max_epochs), bias(0.0) {}

    static double sigmoid(double z) {
        return 1.0 / (1.0 + exp(-z));
    }

    void fit(const vector<vector<double>>& X, const vector<int>& y) {
        int n_samples = X.size();
        int n_features = X[0].size();
        weights.assign(n_features, 0.0);
        bias = 0.0;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int i = 0; i < n_samples; ++i) {
                double linear_model = inner_product(X[i].begin(), X[i].end(), weights.begin(), 0.0) + bias;
                double y_pred = sigmoid(linear_model);
                double error = y_pred - y[i];

                for (int j = 0; j < n_features; ++j) {
                    weights[j] -= lr * error * X[i][j];
                }
                bias -= lr * error;
            }
        }
    }

    int predict(const vector<double>& x) {
        double linear_model = inner_product(x.begin(), x.end(), weights.begin(), 0.0) + bias;
        return sigmoid(linear_model) >= 0.5 ? 1 : 0;
    }
};

// Helper to predict batch for evaluation
inline vector<int> predict_logreg_batch(LogisticRegression &model, const vector<vector<double>>& X) {
    vector<int> preds;
    for (auto &x : X) {
        preds.push_back(model.predict(x));
    }
    return preds;
}

#endif // LOGISTIC_REGRESSION_HPP
