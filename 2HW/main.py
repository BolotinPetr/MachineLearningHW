import pydotplus
import pandas as pd
import math
from pydotplus import graphviz
from IPython.display import Image
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.datasets import load_boston
from tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('german_credit.csv')

target = data[data.columns[0]]
train = data[data.columns[1:]]
m_d = 7
boston = load_boston()
X, X_test, y, y_test = train_test_split(boston.data, boston.target, test_size=0.25)
model = DecisionTreeClassifier(max_depth=m_d)
print X[:10]
model.fit(X, y)
# a = []
print y_test[:10]
# for i in range(0, y.shape[0], 1):
#     a.append(model.predict(X[i]))
# print a[:10]
a = model.predict(X_test)
print a[:10]
print math.sqrt(np.sum((y_test-a)**2)/float(len(a)))

model2 = DecisionTreeRegressor(max_depth=m_d)
model2.fit(X, y)
b = model2.predict(X_test)
#print model2
print b[:10]
print math.sqrt(np.sum((y_test-b)**2)/float(len(b)))


