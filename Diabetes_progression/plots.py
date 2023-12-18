import matplotlib.pyplot as plt
import numpy as np


def model_plots(mse_values, mae_values):
    models_list = list(mse_values.keys())

    mse_values = list(mse_values.values())
    mae_values = list(mae_values.values())

    font1 = {"size": "10"}
    font2 = {"color": "green", "size": "15", "weight": "bold"}
    font3 = {"color": "black", "size": "15", "weight": "bold"}
    font4 = {"color": "white", "weight": "bold"}

    counter = 0
    colors = []
    while counter < len(models_list):
        colors = np.append(colors, "orange")
        counter += 1
        colors = np.append(colors, "blue")
        counter += 1

    def add_text(y):
        for i in range(len(y)):
            plt.text(i, y[i] / 2, y[i], ha="center", fontdict=font4)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    indexes = np.arange(len(models_list))

    plt.subplot(1, 2, 1)
    plt.bar(indexes, mse_values, color=colors)
    plt.xlabel("regression_model", fontdict=font2)
    plt.ylabel("mean_sqaure_error", fontdict=font2)
    plt.xticks(indexes, models_list, fontdict=font1)
    plt.title("Meas Square Error", fontdict=font3)

    add_text(mse_values)

    plt.subplot(1, 2, 2)
    plt.bar(indexes, mae_values, color=colors)
    plt.xlabel("regression_model", fontdict=font2)
    plt.ylabel("mean_absolute_error", fontdict=font2)
    plt.xticks(indexes, models_list, fontdict=font1)
    plt.title("Meas Absolute Error", fontdict=font3)

    add_text(mae_values)

    plt.show()
