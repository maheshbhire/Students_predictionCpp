#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <limits>

using namespace std;

struct TreeNode {
    int featureIndex;
    double threshold;
    int prediction;
    TreeNode* left;
    TreeNode* right;

    TreeNode() : featureIndex(-1), threshold(0), prediction(-1), left(nullptr), right(nullptr) {}
};

class DecisionTree {
    TreeNode* root;
    int maxDepth;

public:
    DecisionTree(int max_depth = 5) : root(nullptr), maxDepth(max_depth) {}

    ~DecisionTree() { freeTree(root); }

    void fit(const vector<vector<double>>& X, const vector<int>& y) {
        root = buildTree(X, y, 0);
    }

    int predict(const vector<double>& x) {
        return predictNode(root, x);
    }

    vector<int> predict(const vector<vector<double>>& X) {
        vector<int> preds;
        for (auto& x : X) preds.push_back(predict(x));
        return preds;
    }

private:
    void freeTree(TreeNode* node) {
        if (!node) return;
        freeTree(node->left);
        freeTree(node->right);
        delete node;
    }

    double giniImpurity(const vector<int>& labels) {
        map<int, int> counts;
        for (auto& label : labels) counts[label]++;
        double impurity = 1.0;
        int total = labels.size();
        for (auto& p : counts) {
            double prob = (double)p.second / total;
            impurity -= prob * prob;
        }
        return impurity;
    }

    pair<vector<int>, vector<int>> split(const vector<vector<double>>& X, const vector<int>& y, int featureIndex, double threshold) {
        vector<int> leftIdx, rightIdx;
        for (int i = 0; i < X.size(); ++i) {
            if (X[i][featureIndex] < threshold)
                leftIdx.push_back(i);
            else
                rightIdx.push_back(i);
        }
        return {leftIdx, rightIdx};
    }

    TreeNode* buildTree(const vector<vector<double>>& X, const vector<int>& y, int depth) {
        TreeNode* node = new TreeNode();

        map<int, int> count;
        for (auto& label : y) count[label]++;
        int majorityLabel = max_element(count.begin(), count.end(),
            [](auto& a, auto& b) { return a.second < b.second; })->first;

        if (depth >= maxDepth || y.size() <= 1) {
            node->prediction = majorityLabel;
            return node;
        }

        double bestGini = numeric_limits<double>::max();
        int bestFeature = -1;
        double bestThreshold = 0;
        vector<int> bestLeft, bestRight;

        for (int f = 0; f < X[0].size(); ++f) {
            vector<double> feature_values;
            for (int i = 0; i < X.size(); ++i) feature_values.push_back(X[i][f]);
            sort(feature_values.begin(), feature_values.end());

            for (auto threshold : feature_values) {
                auto [leftIdx, rightIdx] = split(X, y, f, threshold);
                if (leftIdx.empty() || rightIdx.empty()) continue;

                vector<int> leftLabels, rightLabels;
                for (auto idx : leftIdx) leftLabels.push_back(y[idx]);
                for (auto idx : rightIdx) rightLabels.push_back(y[idx]);

                double giniLeft = giniImpurity(leftLabels);
                double giniRight = giniImpurity(rightLabels);

                double weightedGini = (leftLabels.size() * giniLeft + rightLabels.size() * giniRight) / y.size();

                if (weightedGini < bestGini) {
                    bestGini = weightedGini;
                    bestFeature = f;
                    bestThreshold = threshold;
                    bestLeft = leftIdx;
                    bestRight = rightIdx;
                }
            }
        }

        if (bestFeature == -1) {
            node->prediction = majorityLabel;
            return node;
        }

        vector<vector<double>> X_left, X_right;
        vector<int> y_left, y_right;
        for (auto idx : bestLeft) {
            X_left.push_back(X[idx]);
            y_left.push_back(y[idx]);
        }
        for (auto idx : bestRight) {
            X_right.push_back(X[idx]);
            y_right.push_back(y[idx]);
        }

        node->featureIndex = bestFeature;
        node->threshold = bestThreshold;
        node->left = buildTree(X_left, y_left, depth + 1);
        node->right = buildTree(X_right, y_right, depth + 1);
        return node;
    }

    int predictNode(TreeNode* node, const vector<double>& x) {
        if (node->prediction != -1) return node->prediction;
        if (x[node->featureIndex] < node->threshold)
            return predictNode(node->left, x);
        else
            return predictNode(node->right, x);
    }
};

#endif // DECISION_TREE_HPP