import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence
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
from plots import model_plots, plot_coefficients_linear_regression

# Read dataset
dataset = load_diabetes()
X = dataset.data
y = dataset.target

# Data description
dataset_discription = dataset.DESCR
dataset_features = dataset.feature_names

# Convert X and Y to DataFrame
data = pd.DataFrame(data=X, columns=dataset_features)

scaler = MinMaxScaler()
y = np.reshape(y, (len(y), 1))
y = scaler.fit_transform(y)

data["target"] = y


# Check the data type of the columns and if there is any null value
# print(data.info())

# Column correlations
data_corr = data.corr()

# Check the correlation among columns
# sns.pairplot(data, corner=True)
# plt.show()

X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
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
    models_list, X_train, y_train, X_test, y_test
)

# Plot MSE and MAE for different models, also plot coefficient magnitudes for Linear regression
model_plots(
    mean_sqaure_vlaues,
    mean_absolute_values,
    linear_regression_model,
    X_train,
    y_train,
    dataset_features,
)
