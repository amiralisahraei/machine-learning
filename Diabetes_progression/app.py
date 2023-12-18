import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from models import (
    linear_regression_model,
    models_evaluation,
    ridge_regression_model,
    neural_network_model,
    lasso_regression_model,
    polynomial_regression_model,
    random_forest_regression_model,
)
from plots import model_plots

# Read dataset
dataset = load_diabetes()
X = dataset.data
Y = dataset.target

# Data description
dataset_discription = dataset.DESCR
dataset_features = dataset.feature_names

# Convert X and Y to DataFrame
data = pd.DataFrame(data=X, columns=dataset_features)

scaler = MinMaxScaler()
Y = np.reshape(Y, (len(Y), 1))
Y = scaler.fit_transform(Y)

data["target"] = Y


# Check the data type of the columns and if there is any null value
# print(data.info())

# Column correlations
data_corr = data.corr()

# Check the correlation among columns
# sns.pairplot(data, corner=True)
# plt.show()

X = data.iloc[:, 0:-1]
Y = data.iloc[:, -1]

# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)


models_list = {
    "Linear": linear_regression_model,
    "Polynomial": polynomial_regression_model,
    "Ridge": ridge_regression_model,
    "Lasso": lasso_regression_model,
    "Random\nForest": random_forest_regression_model,
    "Neural\nNetwork": neural_network_model,
}


mean_sqaure_vlaues, mean_absolute_values = models_evaluation(
    models_list, X_train, Y_train, X_test, Y_test
)

model_plots(mean_sqaure_vlaues, mean_absolute_values)
