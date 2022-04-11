import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score  # For find accuracy with R2 Score
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("Regression Dataset/Volumetric_features.csv")

data.info(verbose=False)

X = data.drop(["Age"], axis=1)
Y = data.Age.values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

regr = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))

regr.fit(x_train, y_train)

y_pred_SGD_train = regr.predict(x_train)
y_pred_SGD_test = regr.predict(x_test)

accuracy_SGD_train = r2_score(y_train, y_pred_SGD_train)
print("Training Accuracy for Multiple Linear Regression Model with SGD: ", accuracy_SGD_train)

accuracy_SGD_test = r2_score(y_test, y_pred_SGD_test)
print("Testing Accuracy for Multiple Linear Regression Model with SGD: ", accuracy_SGD_test)

RMSE_SGD_train = sqrt(mean_squared_error(y_train, y_pred_SGD_train))
print("RMSE for Training Data: ", RMSE_SGD_train)

RMSE_SGD_test = sqrt(mean_squared_error(y_test, y_pred_SGD_test))
print("RMSE for Testing Data: ", RMSE_SGD_test)
p = sns.scatterplot(y_train, y_pred_SGD_train)
p.set(xlabel="Actual Age", ylabel="Predicted Age")
plt.show()
