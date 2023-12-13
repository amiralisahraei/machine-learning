import matplotlib.pyplot as plt
import numpy as np


# Show the scores on the top of the  bars


def subplot_models(
    list_of_models, list_of_vlaues, list_of_y_labels, list_of_titles, colors, bar_width
):
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    font1 = {"color": "green", "size": "15", "weight": "bold"}
    font2 = {"color": "white", "size": "12", "weight": "bold"}
    font3 = {"color": "Black", "size": "15", "weight": "bold"}

    def addlabels(x, y):
        for i in range(len(x)):
            plt.text(i, y[i] / 2, y[i], ha="center", fontdict=font2)

    for i in range(0, 4):
        plt.subplot(2, 2, i + 1)
        plt.bar(list_of_models, list_of_vlaues[i], color=colors, width=bar_width)

        addlabels(list_of_models, list_of_vlaues[i])

        plt.xlabel("Models", fontdict=font1)
        plt.ylabel(list_of_y_labels[i], fontdict=font1)
        plt.title(list_of_titles[i], fontdict=font3)

    plt.show()


def models_evaluation_measures(accuracy, precsion, recall, f1):
    list_of_models = list(accuracy.keys())

    def extract_values_from_dict(target_object):
        values = list(target_object.values())
        values = np.around(values, 2)

        return values

    list_of_accuracies = extract_values_from_dict(accuracy)
    list_of_precisions = extract_values_from_dict(precsion)
    list_of_recalls = extract_values_from_dict(recall)
    list_of_f1 = extract_values_from_dict(f1)

    list_of_all_measures = np.vstack(
        (list_of_accuracies, list_of_precisions, list_of_recalls, list_of_f1)
    )
    colors = np.array(["orange", "blue", "orange", "blue", "orange", "blue"])
    y_labels = ["accuracy_score", "precision_score", "recall_score", "f1_score"]
    titles = ["Accuracy", "Precision", "Recall", "F1"]

    subplot_models(list_of_models, list_of_all_measures, y_labels, titles, colors, 0.5)
