
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


### Decision tree structure 

## Load data
X = 'data'
y = 'data'

## what type of data does train_test_split take? #Split arrays or matrices into random train and test subsets

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

estimator = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
estimator.fit(X_train, y_train)