import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from models import (
    decision_tree_model,
    svm_model,
    logistic_regression_model,
    knn_model,
    models_score,
    Nural_Networl_func,
    xgboost_model,
)
from pre_processing import normalization_data, onehot_encoding
from plots import plot_models_evaluation

data_path = "/home/amirali/Amirali/Practice_Python/Breast_cancer_diagnosis/Breast_cancer_data.csv"
data = pd.read_csv(data_path)

# Data info
# data_information = data.info()
# print(data_information)

# Delete useless columns
data.drop("Unnamed: 32", axis=1, inplace=True)
data.drop("id", axis=1, inplace=True)

# Separate X and Y
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# Check all y data are String
for index, i in y.items():
    if not isinstance(i, str):
        print(f"There is a non-string data at index: {index}")


# Check all X data are float
for col in X.columns:
    for index, x in X[col].items():
        if not isinstance(x, float):
            print(f"There is a non-float value in columns: {col} at index: {index}")


# Train and Test split
def test_train_split_data(X, y, testSize):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=testSize, random_state=42, shuffle=True
    )

    return X_train, X_test, Y_train, Y_test


# Plot distribution of Labels (M, B) in train and test data
def distribution_plot(Y_train, Y_test):
    plt.subplot(1, 2, 1)
    plt.hist(Y_train)
    plt.title("Train Labels")

    plt.subplot(1, 2, 2)
    plt.hist(Y_test)
    plt.title("Test Labels")

    plt.show()


def train_models_and_evaluate(X, y, balancing):
    X_train, X_test, Y_train, Y_test = test_train_split_data(X, y, 0.2)

    if balancing:
        X_train, Y_train = SMOTE().fit_resample(X_train, Y_train)

    X_train = normalization_data(X_train)
    X_test = normalization_data(X_test)

    Y_train = onehot_encoding(Y_train)
    Y_test = onehot_encoding(Y_test)

    models_list = [
        decision_tree_model,
        svm_model,
        logistic_regression_model,
        knn_model,
        xgboost_model,
        Nural_Networl_func,
    ]

    measure_values = models_score(models_list, X_train, Y_train, X_test, Y_test)

    return measure_values


# Imbalanced Dataset
measure_values = train_models_and_evaluate(X, y, False)

# Balanced Dataset
measure_values2 = train_models_and_evaluate(X, y, True)


# Merge the measures resulted from Imbalanced and Balanced dataset
def merge_arrays(measure_values, measure_values2):
    list_of_measures = []
    for i in range(len(measure_values)):
        merged_array = np.stack((measure_values[i], measure_values2[i]))
        list_of_measures.append(merged_array)

    list_of_measures = np.array(list_of_measures)

    return list_of_measures


measure_values = merge_arrays(measure_values, measure_values2)


list_of_models = np.array(
    ["decision_tree", "SVM", "Logistic_Regression", "KNN", "XGBoost", "Neural Network"]
)
y_labels = ["roc_auc_score", "precision_score", "recall_score", "f1_score"]
titles = ["ROC_AUC", "Precision", "Recall", "F1"]


plot_models_evaluation(list_of_models, measure_values, titles, y_labels, 0.35)
