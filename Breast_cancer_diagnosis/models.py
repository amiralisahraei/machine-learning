import numpy as np
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import tensorflow as tf
from keras import models, layers


# Decision Tree model
def decision_tree_model(X_train, y_train):
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)

    return decision_tree


# SVM model
def svm_model(X_train, y_train):
    svm_model = svm.SVC(kernel="linear", C=1.0, random_state=42)
    svm_model.fit(X_train, y_train)

    return svm_model


# Logistic Regression
def logistic_regression_model(X_train, y_train):
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)

    return logistic_model


# KNN model
def knn_model(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, y_train)

    return knn


# XGBoost model
def xgboost_model(X_train, y_train):
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)

    return xgb


# Neural Network model
def Nural_Networl_func(X_train, y_train):
    neural_network_model = models.Sequential(
        [
            layers.Dense(64, activation="tanh", input_shape=(X_train.shape[1],)),
            layers.Dense(32, activation="tanh"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    neural_network_model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    neural_network_model.fit(
        X_train, y_train, epochs=10, batch_size=32, validation_split=0.2
    )

    return neural_network_model


# Model prediction
def model_prediction(model_name, X_train, y_train, X_test, y_test):
    model = model_name(X_train, y_train)
    predicted_labels = model.predict(X_test)
    # Bcause the output of Neural Network is not 1 or 0 we need to convert it to a binary array
    predicted_labels = (predicted_labels > 0.5).astype("int")

    roc_auc_value = roc_auc_score(y_test, predicted_labels)
    precision_value = precision_score(y_test, predicted_labels)
    recall_value = recall_score(y_test, predicted_labels)
    f1_value = f1_score(y_test, predicted_labels)

    return roc_auc_value, precision_value, recall_value, f1_value


# Count evaluation measures for different models
def models_score(models_list, X_train, y_train, X_test, y_test):
    roc_auc_values = []
    precision_values = []
    recall_values = []
    f1_values = []
    for model_func in models_list:
        roc_auc, precision, recall, f1 = model_prediction(
            model_func, X_train, y_train, X_test, y_test
        )

        roc_auc_values = np.append(roc_auc_values, np.around(roc_auc, 2))
        precision_values = np.append(precision_values, np.around(precision, 2))
        recall_values = np.append(recall_values, np.around(recall, 2))
        f1_values = np.append(f1_values, np.around(f1, 2))

    result = np.stack((roc_auc_values, precision_values, recall_values, f1_values))

    return result
