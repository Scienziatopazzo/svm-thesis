import pandas as pd
import numpy as np
import itertools as it
from sklearn.model_selection import train_test_split

#Grid search model selection for SVR with holdout. Needs a param_grid with all possible parameters stated, even ones not needed for a particular kernel
def SVR_gridsearch_holdout(X, y, estimator, param_grid, test_size, val_size, scaler=None, outlier_detector=None):
    X_TrainAndValidation, X_Test, y_TrainAndValidation, y_Test = train_test_split(X, y.astype('float64'), test_size=test_size)
    X_Train, X_Validation, y_Train, y_Validation = train_test_split(X_TrainAndValidation, y_TrainAndValidation, test_size=val_size)

    #Scale training set
    if scaler is not None:
        X_Scaler, y_Scaler = scaler(), scaler()
        X_Scaler.fit(X_Train)
        y_Scaler.fit(y_Train.reshape((-1,1)))
        X_Train, y_Train = X_Scaler.transform(X_Train), y_Scaler.transform(y_Train.reshape((-1,1))).ravel()
        X_Test, y_Test = X_Scaler.transform(X_Test), y_Scaler.transform(y_Test.reshape((-1,1))).ravel()
        X_Validation, y_Validation = X_Scaler.transform(X_Validation), y_Scaler.transform(y_Validation.reshape((-1,1))).ravel()
        X_TrainAndValidation, y_TrainAndValidation = X_Scaler.transform(X_TrainAndValidation), y_Scaler.transform(y_TrainAndValidation.reshape((-1,1))).ravel()

    best_score = -np.inf
    best_params = (0,0,0,0,0,0)

    for grid in param_grid:
        allParams = sorted(grid)
        combinations = it.product(*(grid[i] for i in allParams))

        for C, coef0, degree, epsilon, gamma, kernel in combinations:
            svr = estimator(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, C=C, epsilon=epsilon)
            svr.fit(X_Train, y_Train)
            score = svr.score(X_Validation, y_Validation)
            if score > best_score:
                best_score = score
                best_params = {'C':C, 'coef0':coef0, 'degree':degree, 'epsilon':epsilon, 'gamma':gamma, 'kernel':kernel}

    best_svr = estimator(**best_params)
    best_svr.fit(X_TrainAndValidation, y_TrainAndValidation)

    test_score = best_svr.score(X_Test, y_Test)

    return (best_params, test_score)
