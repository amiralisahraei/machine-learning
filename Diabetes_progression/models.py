import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras import models, layers
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# Linear regression
def linear_regression_model(X_train, y_train):
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    return linear_model


# Polynomila regression
def polynomial_regression_model(X_train, y_train, degree=2):
    ploynomial_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    ploynomial_model.fit(X_train, y_train)

    return ploynomial_model


# Ridge regression
def ridge_regression_model(X_train, y_train):
    ridge_model = Ridge(random_state=42)
    ridge_model.fit(X_train, y_train)

    return ridge_model


# Lasso regression
def lasso_regression_model(X_train, y_train):
    lasso_model = Lasso(random_state=42)
    lasso_model.fit(X_train, y_train)

    return lasso_model


# Random Forest regression
def random_forest_regression_model(X_train, y_train):
    rand_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rand_forest_model.fit(X_train, y_train)

    return rand_forest_model


# Neural Network regression
def neural_network_model(X_train, y_train):
    neural_model = models.Sequential(
        [
            layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="linear"),
        ]
    )

    neural_model.compile(
        optimizer="adam", loss="mean_squared_error", metrics=["accuracy"]
    )
    neural_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    return neural_model


# Evalaute the models 
def models_evaluation(models_list, X_train, y_train, X_test, y_test):
    mse_scores = {}
    mae_scores = {}

    for model_name, model_func in models_list.items():
        trained_model = model_func(X_train, y_train)
        predicted_labels = trained_model.predict(X_test)

        mse_score = mean_squared_error(y_test, predicted_labels)
        mse_score = np.around(mse_score, 3)
        mse_scores[model_name] = mse_score

        mae_score = mean_squared_error(y_test, predicted_labels)
        mae_score = np.around(mae_score, 3)
        mae_scores[model_name] = mae_score

    return (mse_scores, mae_scores)
