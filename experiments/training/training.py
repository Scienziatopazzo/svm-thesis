import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from multiprocessing import Process, Queue
from queue import Empty

#Grid search model selection for SVR with holdout.
def SVR_gridsearch_holdout(X, y, estimator, param_grid, test_size, val_size, scaler=None, outlier_detector=None, nprocs=8, censSVR=False):
    X_TrainAndValidation, X_Test, y_TrainAndValidation, y_Test = train_test_split(X, y.astype('float64'), test_size=test_size)
    X_Train, X_Validation, y_Train, y_Validation = train_test_split(X_TrainAndValidation, y_TrainAndValidation, test_size=val_size)

    #Do outlier detection on training set
    if outlier_detector is not None:
        y_outliers = outlier_detector.fit_predict(np.concatenate((X_Train, y_Train.reshape((-1,1))), axis=1))
        X_Train = X_Train[y_outliers >= 1]
        y_Train = y_Train[y_outliers >= 1]

    #Scale training set
    if scaler is not None:
        X_Scaler, y_Scaler = scaler(), scaler()
        if censSVR:
            X_TrainDeltas = X_Train[:,-1]
            X_Train = X_Train[:,:-1]
            X_ValDeltas = X_Validation[:,-1]
            X_Validation = X_Validation[:,:-1]
        X_Train, y_Train = X_Scaler.fit_transform(X_Train), y_Scaler.fit_transform(y_Train.reshape((-1,1))).ravel()
        X_Validation, y_Validation = X_Scaler.transform(X_Validation), y_Scaler.transform(y_Validation.reshape((-1,1))).ravel()
        if censSVR:
            X_Train = np.concatenate((X_Train,X_TrainDeltas.reshape((-1,1))), axis=1)
            X_Validation = np.concatenate((X_Validation,X_ValDeltas.reshape((-1,1))), axis=1)

    best_score = -np.inf
    best_params = {}
    b_queue = Queue()
    p_total_num = 0

    for params in ParameterGrid(param_grid):
        while p_total_num >= nprocs:
            score, fit_params = b_queue.get()
            while p_total_num > 0:
                if score > best_score:
                    best_score = score
                    best_params = fit_params
                p_total_num -= 1
                try:
                    score, fit_params = b_queue.get(block=False)
                except Empty:
                    break

        p_total_num += 1
        p = Process(target=proc_train, args=(b_queue, estimator, X_Train, y_Train, X_Validation, y_Validation, params))
        p.start()

    while p_total_num > 0:
        score, fit_params = b_queue.get()
        if score > best_score:
            best_score = score
            best_params = fit_params
        p_total_num -= 1

    best_svr = estimator(**best_params)

    #Redo both outlier detection and scaling on joined train and validation sets with final parameters
    if outlier_detector is not None:
        y_outliers = outlier_detector.fit_predict(np.concatenate((X_TrainAndValidation, y_TrainAndValidation.reshape((-1,1))), axis=1))
        X_TrainAndValidation = X_TrainAndValidation[y_outliers >= 1]
        y_TrainAndValidation = y_TrainAndValidation[y_outliers >= 1]

    if scaler is not None:
        if censSVR:
            X_TrainDeltas = X_TrainAndValidation[:,-1]
            X_TrainAndValidation = X_TrainAndValidation[:,:-1]
            X_TestDeltas = X_Test[:,-1]
            X_Test = X_Test[:,:-1]
        X_TrainAndValidation, y_TrainAndValidation = X_Scaler.fit_transform(X_TrainAndValidation), y_Scaler.fit_transform(y_TrainAndValidation.reshape((-1,1))).ravel()
        X_Test, y_Test = X_Scaler.transform(X_Test), y_Scaler.transform(y_Test.reshape((-1,1))).ravel()
        if censSVR:
            X_TrainAndValidation = np.concatenate((X_TrainAndValidation,X_TrainDeltas.reshape((-1,1))), axis=1)
            X_Test = np.concatenate((X_Test,X_TestDeltas.reshape((-1,1))), axis=1)

    best_svr.fit(X_TrainAndValidation, y_TrainAndValidation)
    test_score = best_svr.score(X_Test, y_Test)

    return (best_params, test_score)

def proc_train(b_queue, estimator, X_Train, y_Train, X_Validation, y_Validation, fit_params):
    svr = estimator(**fit_params)
    svr.fit(X_Train, y_Train)
    score = svr.score(X_Validation, y_Validation)
    b_queue.put((score, fit_params))

#Finally testing a parameter set for an estimator on random test splits
def random_split_tests(X, y, estimator, params, test_size, ntests=10, scaler=None, outlier_detector=None, censSVR=False):
    mean_score = 0
    for i in range(ntests):
        X_Train, X_Test, y_Train, y_Test = train_test_split(X, y.astype('float64'), test_size=test_size)
        if outlier_detector is not None:
            y_outliers = outlier_detector.fit_predict(np.concatenate((X_Train, y_Train.reshape((-1,1))), axis=1))
            X_Train = X_Train[y_outliers >= 1]
            y_Train = y_Train[y_outliers >= 1]
        if scaler is not None:
            X_Scaler, y_Scaler = scaler(), scaler()
            if censSVR:
                X_TrainDeltas = X_Train[:,-1]
                X_Train = X_Train[:,:-1]
                X_TestDeltas = X_Test[:,-1]
                X_Test = X_Test[:,:-1]
            X_Scaler.fit(X_Train)
            y_Scaler.fit(y_Train.reshape((-1,1)))
            X_Train, y_Train = X_Scaler.transform(X_Train), y_Scaler.transform(y_Train.reshape((-1,1))).ravel()
            X_Test, y_Test = X_Scaler.transform(X_Test), y_Scaler.transform(y_Test.reshape((-1,1))).ravel()
            if censSVR:
                X_Train = np.concatenate((X_Train,X_TrainDeltas.reshape((-1,1))), axis=1)
                X_Test = np.concatenate((X_Test,X_TestDeltas.reshape((-1,1))), axis=1)

        best_svr = estimator(**params)
        best_svr.fit(X_Train, y_Train)
        mean_score += best_svr.score(X_Test, y_Test)/ntests
    return mean_score
