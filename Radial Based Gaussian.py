import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

data = pd.read_csv("ClassificationDataset/wdbcgood.csv")

data.info(verbose=False)

sns.catplot(x="diagnosis", kind="count", palette="Set1", data=data)


data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

X = data.drop(["diagnosis"], axis=1)
Y = data.diagnosis.values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

kernel = 1.0 * RBF(10)
clf = gpc = GaussianProcessClassifier(kernel=kernel, random_state=5)
clf.fit(x_train, y_train)

y_pred_SVM_train = clf.predict(x_train)
y_pred_SVM_test = clf.predict(x_test)

accuracy_SVM_train = r2_score(y_train, y_pred_SVM_train)
print("Training Accuracy for Gaussian RBF Model: ", accuracy_SVM_train)

accuracy_SVM_test = r2_score(y_test, y_pred_SVM_test)
print("Testing Accuracy for Gaussian RBF Model: ", accuracy_SVM_test)


def confusion_matrix(true, pred):
    K = len(np.unique(true))  # Number of classes
    result = np.zeros((K, K))
    for i in range(len(true)):
        result[true[i]][pred[i]] += 1
    return result


con_mtx = confusion_matrix(y_test, y_pred_SVM_test)
con_df = pd.DataFrame(con_mtx, columns=['Benign', 'Malignant'], index=['Benign_pred', 'Malignant_pred'])
sns.heatmap(con_df, annot=True, cbar=False)
plt.show()
print(con_mtx)

