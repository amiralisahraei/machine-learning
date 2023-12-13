from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from keras import models, layers


# Decision Tree model
def decision_tree_model(X_train, Y_train):
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, Y_train)

    return decision_tree


# SVM model
def svm_model(X_train, Y_train):
    svm_model = svm.SVC(kernel="linear", C=1.0, random_state=42)
    svm_model.fit(X_train, Y_train)

    return svm_model


# Logistic Regression
def logistic_regression_model(X_train, Y_train):
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, Y_train)

    return logistic_model


# KNN model
def knn_model(X_train, Y_train):
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, Y_train)

    return knn


# XGBoost model
def xgboost_model(X_train, Y_train):
    xgb = XGBClassifier()
    xgb.fit(X_train, Y_train)

    return xgb


# Neural Network model
def Nural_Networl_func(X_train, Y_train):
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
        X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2
    )

    return neural_network_model



# Model prediction
def model_prediction(model_name, X_train, Y_train, X_test, Y_test):
    model = model_name(X_train, Y_train)
    predicted_labels = model.predict(X_test)
    # Bcause the output of Neural Network is not 1 or 0 we need to convert it to a binary array
    predicted_labels = (predicted_labels > 0.5).astype("int")

    accuracy_value = accuracy_score(Y_test, predicted_labels)
    precision_value = precision_score(Y_test, predicted_labels)
    recall_value = recall_score(Y_test, predicted_labels)
    f1_value = f1_score(Y_test, predicted_labels)

    return accuracy_value, precision_value, recall_value, f1_value


# Accuracy score for different models
def models_accuracy_score(models_list, X_train, Y_train, X_test, Y_test):
    accuracy_vlaues = {}
    precision_values = {}
    recall_values = {}
    f1_values = {}
    for model_name, model_func in models_list.items():
        accuracy, precision, recall, f1 = model_prediction(
            model_func, X_train, Y_train, X_test, Y_test
        )

        accuracy_vlaues[model_name] = accuracy
        precision_values[model_name] = precision
        recall_values[model_name] = recall
        f1_values[model_name] = f1

    return accuracy_vlaues, precision_values, recall_values, f1_values
