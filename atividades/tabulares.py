from pprint import pprint

import pandas as pd
import numpy as np
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
X = data.data
y:Series = data.target
l = list(X.columns)
pprint(l)

print(np.sum(y) / y.shape[0])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
