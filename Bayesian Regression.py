
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score  # For find accuracy with R2 Score
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn import linear_model


data = pd.read_csv("Regression Dataset/Volumetric_features.csv")

data.info(verbose=False)

X = data.drop(["Age"], axis=1)
Y = data.Age.values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

reg = linear_model.BayesianRidge()
reg.fit(x_train, y_train)

y_pred_B_train = reg.predict(x_train)
y_pred_B_test = reg.predict(x_test)

accuracy_B_train = r2_score(y_train, y_pred_B_train)
print("Training Accuracy for Bayesian Regression Model: ", accuracy_B_train)

accuracy_B_test = r2_score(y_test, y_pred_B_test)
print("Testing Accuracy for Bayesian Regression Model: ", accuracy_B_test)

RMSE_B_train = sqrt(mean_squared_error(y_train, y_pred_B_train))
print("RMSE for Training Data: ", RMSE_B_train)

RMSE_B_test = sqrt(mean_squared_error(y_test, y_pred_B_test))
print("RMSE for Testing Data: ", RMSE_B_test)

p = sns.scatterplot(y_train, y_pred_B_train)
p.set(xlabel="Actual Age", ylabel="Predicted Age")
plt.show()

