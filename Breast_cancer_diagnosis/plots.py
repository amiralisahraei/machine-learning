import matplotlib.pyplot as plt
import numpy as np


def plot_models_evaluation(
    list_of_models, measure_values, list_of_titles, list_of_y_labels, bar_width
):
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    font1 = {"color": "green", "size": "15", "weight": "bold"}
    font2 = {"color": "white", "size": "9", "weight": "bold"}
    font3 = {"color": "Black", "size": "15", "weight": "bold"}

    def addlabels(x, y):
        for i in range(len(x)):
            plt.text(i, y[i] / 2, y[i], ha="center", fontdict=font2)

    def addlabels2(x, y):
        for i in range(len(x)):
            plt.text(
                i + (bar_width + 0.05), y[i] / 2, y[i], ha="center", fontdict=font2
            )

    for i in range(0, 4):
        plt.subplot(2, 2, i + 1)
        index = np.arange(len(list_of_models))  # Bar positions

        plt.bar(
            index,
            measure_values[i][0],
            color="blue",
            width=bar_width,
            label="Imbalanced",
        )
        plt.bar(
            index + (bar_width + 0.05),
            measure_values[i][1],
            color="orange",
            width=bar_width,
            label="Balanced",
        )

        addlabels(list_of_models, measure_values[i][0])
        addlabels2(list_of_models, measure_values[i][1])

        plt.xlabel("Models", fontdict=font1)
        plt.ylabel(list_of_y_labels[i], fontdict=font1)
        plt.title(list_of_titles[i], fontdict=font3)
        plt.xticks(index + bar_width / 2, list_of_models)
        plt.legend()

    plt.show()
