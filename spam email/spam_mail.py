import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets

from naive_bayes import NaiveBayes 

def accuracy(y_true, y_pred):
    # sourcery skip: inline-immediately-returned-variable
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

train_df=pd.read_csv('train_data.csv')
test_df=pd.read_csv('test_features.csv')

train_df=train_df.drop('Id',axis=1)
test_df=test_df.drop('Id',axis=1)

X=np.asanyarray(train_df.drop('ham',axis=1))
y=np.asanyarray(train_df['ham'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

print(f'Naive Bayes classification accuracy {accuracy(y_test, predictions)}')