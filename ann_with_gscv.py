#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:52:45 2020

@author: ravindersaluja
"""
# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
df = pd.read_csv('Churn_Modelling.csv')
X = df.iloc[:, 3:13]
y = df.iloc[:, 13]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Dummy Variable Generation
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
column_trans = make_column_transformer((OneHotEncoder(drop='first'), ['Geography', 'Gender'])
                                       , remainder='passthrough')
X_train = column_trans.fit_transform(X_train)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

# Transforming the test data as well
X_test = column_trans.transform(X_test)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score

# Dropout regularlization to prevent overfitting
# from keras.layers import Dropout

# Tuning the ANN
from sklearn.model_selection import GridSearchCV


def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
cvResults = pd.DataFrame(grid_search.cv_results_)