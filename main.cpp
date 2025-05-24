#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include "encoder.hpp"
#include "decision_tree.hpp"
#include "logistic_regression.hpp"
#include "utils.hpp"
#include "svm.hpp"
#include "knn.hpp"
#include "mlp.hpp"

using namespace std;

int main()
{
    string filename = "AI-Data.csv";
    vector<vector<string>> rawData = loadCSV(filename);

    if (rawData.empty())
    {
        cerr << "Failed to load data or empty dataset." << endl;
        return 1;
    }

    vector<vector<double>> X;
    vector<int> y;
    vector<map<string, int>> encoders;

    preprocessData(rawData, X, y, encoders);

    cout << "Dataset loaded and preprocessed." << endl;
    cout << "Number of samples: " << X.size() << endl;

    // Split data (70-30 split)
    int train_size = int(X.size() * 0.7);
    vector<vector<double>> X_train(X.begin(), X.begin() + train_size);
    vector<int> y_train(y.begin(), y.begin() + train_size);
    vector<vector<double>> X_test(X.begin() + train_size, X.end());
    vector<int> y_test(y.begin() + train_size, y.end());

    // Create models
    DecisionTree dtree;
    LogisticRegression logreg(0.01, 1000);
    SVM svm;
    KNN knn(3);
    MLP mlp;

    // Train models
    dtree.fit(X_train, y_train);
    logreg.fit(X_train, y_train);
    svm.fit(X_train, y_train);
    knn.fit(X_train, y_train);
    mlp.fit(X_train, y_train);

    // Evaluate models
    cout << "\nEvaluating models on test data:\n";

    cout << "\nDecision Tree Accuracy: " << accuracy_score(y_test, dtree.predict(X_test)) << endl;
    cout << "Logistic Regression Accuracy: " << accuracy_score(y_test, predict_logreg_batch(logreg, X_test)) << endl;
    cout << "SVM Accuracy: " << accuracy_score(y_test, svm.predict_batch(X_test)) << endl;
    cout << "KNN Accuracy: " << accuracy_score(y_test, knn.predict_batch(X_test)) << endl;
    cout << "MLP Accuracy: " << accuracy_score(y_test, mlp.predict_batch(X_test)) << endl;

    // Menu interaction (simplified)
    int choice;
    cout << "\nEnter 1 to predict with Decision Tree, 2 Logistic Regression, 3 SVM, 4 KNN, 5 MLP, 0 to exit: ";
    while (cin >> choice && choice != 0)
    {
        vector<double> input_features;
        cout << "Enter feature values separated by space (" << X[0].size() << " features): ";
        input_features.resize(X[0].size());
        for (auto &val : input_features)
            cin >> val;

        int pred = -1;
        switch (choice)
        {
        case 1:
            pred = dtree.predict(input_features);
            break;
        case 2:
            pred = logreg.predict(input_features);
            break;
        case 3:
            pred = svm.predict(input_features);
            break;
        case 4:
            pred = knn.predict(input_features);
            break;
        case 5:
            pred = mlp.predict(input_features);
            break;
        default:
            cout << "Invalid choice." << endl;
            continue;
        }

        cout << "Predicted class: " << pred << endl;
        cout << "Enter next choice (1-5) or 0 to exit: ";
    }

    cout << "Exiting..." << endl;
    return 0;
}
