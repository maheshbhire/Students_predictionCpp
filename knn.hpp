#ifndef KNN_HPP
#define KNN_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <map>

using namespace std;

class KNN {
    int k;
    vector<vector<double>> X_train;
    vector<int> y_train;

public:
    KNN(int neighbors = 3) : k(neighbors) {}

    void fit(const vector<vector<double>>& X, const vector<int>& y) {
        X_train = X;
        y_train = y;
    }

    int predict(const vector<double>& x) {
        vector<pair<double, int>> distances; // distance, label

        for (int i = 0; i < X_train.size(); ++i) {
            double dist = 0;
            for (int j = 0; j < x.size(); ++j) {
                dist += (X_train[i][j] - x[j]) * (X_train[i][j] - x[j]);
            }
            distances.push_back({ sqrt(dist), y_train[i] });
        }

        sort(distances.begin(), distances.end(),
             [](pair<double, int> a, pair<double, int> b) { return a.first < b.first; });

        map<int, int> votes;
        for (int i = 0; i < k; ++i) {
            votes[distances[i].second]++;
        }

        int max_vote = 0, pred = -1;
        for (auto& v : votes) {
            if (v.second > max_vote) {
                max_vote = v.second;
                pred = v.first;
            }
        }
        return pred;
    }

    vector<int> predict_batch(const vector<vector<double>>& X) {
        vector<int> preds;
        for (auto& x : X) preds.push_back(predict(x));
        return preds;
    }
};

#endif // KNN_HPP
