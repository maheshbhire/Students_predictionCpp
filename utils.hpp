#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>

using namespace std;

vector<vector<string>> loadCSV(const string& filename) {
    vector<vector<string>> data;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return data;
    }
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string item;
        vector<string> row;
        while (getline(ss, item, ',')) {
            row.push_back(item);
        }
        data.push_back(row);
    }
    file.close();
    return data;
}

double accuracy_score(const vector<int>& y_true, const vector<int>& y_pred) {
    if (y_true.size() != y_pred.size() || y_true.empty()) return 0.0;
    int correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) ++correct;
    }
    return (double)correct / y_true.size();
}

#endif // UTILS_HPP
