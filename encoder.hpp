#ifndef ENCODER_HPP
#define ENCODER_HPP

#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <algorithm>
#include "utils.hpp"

using namespace std;

void preprocessData(const vector<vector<string>>& rawData,
                    vector<vector<double>>& X,
                    vector<int>& y,
                    vector<map<string, int>>& encoders) {
    if (rawData.empty()) return;

    int n_features = rawData[0].size() - 1;
    encoders.resize(n_features);

    X.clear();
    y.clear();

    // Encode categorical features to int
    for (int col = 0; col < n_features; ++col) {
        map<string, int> mapping;
        int code = 0;
        for (int row = 0; row < rawData.size(); ++row) {
            string val = rawData[row][col];
            if (mapping.find(val) == mapping.end()) {
                mapping[val] = code++;
            }
        }
        encoders[col] = mapping;
    }

    for (auto& row : rawData) {
        vector<double> features;
        for (int col = 0; col < n_features; ++col) {
            string val = row[col];
            features.push_back(encoders[col][val]);
        }
        X.push_back(features);
        y.push_back(stoi(row[n_features]));
    }
}

#endif // ENCODER_HPP
