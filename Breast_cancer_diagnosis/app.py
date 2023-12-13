import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from models import (
    decision_tree_model,
    svm_model,
    logistic_regression_model,
    knn_model,
    models_accuracy_score,
    Nural_Networl_func,
    xgboost_model
)
from pre_processing import normalization_data, onehot_encoding
from plots import models_evaluation_measures

data_path = "/home/amirali/Amirali/Practice_Python/Breast_cancer_diagnosis/Breast_cancer_data.csv"
data = pd.read_csv(data_path)

# Data info
# data_information = data.info()
# print(data_information)

# Data describtion
# data_desciption = data.describe()
# print(data_desciption)

# Delete useless columns
data.drop("Unnamed: 32", axis=1, inplace=True)
data.drop("id", axis=1, inplace=True)

# Separate X and Y
X = data.iloc[:, 1:]
Y = data.iloc[:, 0]

# Check all Y data are String
for index, x in Y.items():
    if not isinstance(x, str):
        print(f"There is a non-string data at index: {index}")


# Check all X data are float
for col in X.columns:
    for index, x in X[col].items():
        if not isinstance(x, float):
            print(f"There is a non-float value in columns: {col} at index: {index}")


# Correlatio between columns
# print(X.corr())

# Plot the relationship between few columns
# sns.pairplto(data[['radius_mean', 'area_worst', 'smoothness_worst']], corner=True)
# plt.show()


# Train and Test split
def test_train_split_data(X, Y, testSize):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=testSize, random_state=42, shuffle=True
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


X_train, X_test, Y_train, Y_test = test_train_split_data(X, Y, 0.2)

# distribution_plot(Y_train, Y_test)

X_train = normalization_data(X_train)
X_test = normalization_data(X_test)

Y_train = onehot_encoding(Y_train)
Y_test = onehot_encoding(Y_test)


models_list = {
    "decision_tree": decision_tree_model,
    "SVM": svm_model,
    "Logistic_Regression": logistic_regression_model,
    "KNN": knn_model,
    "XGBoost": xgboost_model,
    "Neural Network": Nural_Networl_func,
}
accuracy_list, precision_list, recall_list, f1_list = models_accuracy_score(
    models_list, X_train, Y_train, X_test, Y_test
)

# Plot evaluation measures
models_evaluation_measures(accuracy_list, precision_list, recall_list, f1_list)


