import matplotlib.pyplot as plt
import numpy as np
from fonts import font1, font2, font3, font4, font5, font5, font6


def add_text(y, fontdict):
    for i in range(len(y)):
        plt.text(i, y[i] / 2, y[i], ha="center", fontdict=fontdict)


def plot_coefficients_linear_regression(
    linear_regression_model, X_train, Y_train, feature_names
):
    regression_model = linear_regression_model(X_train, Y_train)
    coefficients = regression_model.coef_

    coefficients = np.around(coefficients, 3)

    plt.bar(range(len(feature_names)), coefficients)
    add_text(coefficients, font5)

    plt.xlabel("features", fontdict=font2)
    plt.ylabel("coefficients_mangnitude", fontdict=font2)
    plt.xticks(range(len(feature_names)), feature_names, fontdict=font6)
    plt.title("Linear Regression Coefficients", fontdict=font3)


def model_plots(
    mse_values, mae_values, linear_regression_model, X_train, Y_train, feature_names
):
    models_list = list(mse_values.keys())

    mse_values = list(mse_values.values())
    mae_values = list(mae_values.values())

    counter = 0
    colors = []
    while counter < len(models_list):
        colors = np.append(colors, "orange")
        counter += 1
        colors = np.append(colors, "blue")
        counter += 1

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    indexes = np.arange(len(models_list))

    plt.subplot(2, 2, 1)
    plt.bar(indexes, mse_values, color=colors)
    plt.xlabel("regression_model", fontdict=font2)
    plt.ylabel("mean_sqaure_error", fontdict=font2)
    plt.xticks(indexes, models_list, fontdict=font1)
    plt.title("Meas Square Error", fontdict=font3)

    add_text(mse_values, font4)

    plt.subplot(2, 2, 2)
    plt.bar(indexes, mae_values, color=colors)
    plt.xlabel("regression_model", fontdict=font2)
    plt.ylabel("mean_absolute_error", fontdict=font2)
    plt.xticks(indexes, models_list, fontdict=font1)
    plt.title("Meas Absolute Error", fontdict=font3)

    add_text(mae_values, font4)

    plt.subplot(2, 1, 2)
    plot_coefficients_linear_regression(
        linear_regression_model, X_train, Y_train, feature_names
    )

    plt.show()
