

import numpy as np  # Importing NumPy library
import pandas as pd  # Importing Pandas library
import matplotlib.pyplot as plt  # Importing Matplotlib library's "pyplot" module
import seaborn as sns  # Imorting Seaborn library
from sklearn.model_selection import train_test_split
import os
from sklearn.model_selection import cross_val_predict  # For K-Fold Cross Validation
from sklearn.metrics import r2_score  # For find accuracy with R2 Score
from sklearn.metrics import mean_squared_error  # For MSE
from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from math import sqrt  # For squareroot operation

data = pd.read_csv("Regression Dataset/Volumetric_features.csv")

data.info(verbose=False)

#
# data.head()
#
# data.tail()
#
# data.describe()
#
# data.corr()
#
# fig, axes = plt.subplots(figsize=(8, 8))
# sns.heatmap(data=data.corr(), annot=True, linewidths=.5, ax=axes)
# plt.show()

#
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
# data.plot(kind="hist", y="Age", bins=70, color="b", ax=axes[0][0])
# data.plot(kind="hist", y="Left-Hippocampus", bins=200, color="r", ax=axes[0][1])
# plt.show()


X = data.drop(["Age"], axis=1)
Y = data.Age.values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

regr.fit(x_train, y_train)

# multiple_linear_reg = LinearRegression(fit_intercept=False)
# multiple_linear_reg.fit(x_train, y_train)

y_pred_SVM_train = regr.predict(x_train)
y_pred_SVM_test = regr.predict(x_test)

accuracy_SVM_train = r2_score(y_train, y_pred_SVM_train)
print("Training Accuracy for Support Vector Machines: ", accuracy_SVM_train)

accuracy_SVM_test = r2_score(y_test, y_pred_SVM_test)
print("Testing Accuracy for Support Vector Machines: ", accuracy_SVM_test)

RMSE_SVM_train = sqrt(mean_squared_error(y_train, y_pred_SVM_train))
print("RMSE for Training Data: ", RMSE_SVM_train)

RMSE_SVM_test = sqrt(mean_squared_error(y_test, y_pred_SVM_test))
print("RMSE for Testing Data: ", RMSE_SVM_test)

p = sns.scatterplot(y_train, y_pred_SVM_train)
p.set(xlabel="Actual Age", ylabel="Predicted Age")
plt.show()

# print(multiple_linear_reg.coef_)

