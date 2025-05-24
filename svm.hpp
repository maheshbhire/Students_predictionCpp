#ifndef SVM_HPP
#define SVM_HPP

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace std;

class SVM {
    vector<double> weights;
    double bias;
    double lr;
    int epochs;

public:
    SVM(double learning_rate = 0.01, int max_epochs = 1000)
        : lr(learning_rate), epochs(max_epochs), bias(0.0) {}

    void fit(const vector<vector<double>>& X, const vector<int>& y) {
        int n_samples = X.size();
        int n_features = X[0].size();
        weights.assign(n_features, 0.0);
        bias = 0.0;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int i = 0; i < n_samples; ++i) {
                double linear_output = inner_product(X[i].begin(), X[i].end(), weights.begin(), 0.0) + bias;
                int y_i = y[i] == 1 ? 1 : -1; // Convert label to {-1,1}

                if (y_i * linear_output < 1) {
                    for (int j = 0; j < n_features; ++j)
                        weights[j] += lr * (y_i * X[i][j] - 2 * 0.01 * weights[j]);
                    bias += lr * y_i;
                } else {
                    for (int j = 0; j < n_features; ++j)
                        weights[j] += lr * (-2 * 0.01 * weights[j]);
                }
            }
        }
    }

    int predict(const vector<double>& x) {
        double linear_output = inner_product(x.begin(), x.end(), weights.begin(), 0.0) + bias;
        return linear_output >= 0 ? 1 : 0;
    }

    vector<int> predict_batch(const vector<vector<double>>& X) {
        vector<int> preds;
        for (auto& x : X) preds.push_back(predict(x));
        return preds;
    }
};

#endif // SVM_HPP
