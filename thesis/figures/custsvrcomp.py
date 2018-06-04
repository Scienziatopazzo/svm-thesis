#!/usr/bin/python3

import sys
sys.path.append("../../experiments")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import customsvr


from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.neighbors import LocalOutlierFactor
from datasets import *
from training import *

np.random.seed(13)
#linear ex
plt.figure()
plt.title("Linear objective function")

npoints = 50
noise = 0.1
w = np.random.randn(3)

X = np.random.rand(npoints, 2)*2 -1
y = np.matmul(X,w[:2])+w[2] + (np.random.randn(npoints)*noise)

X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size = 1/4)

plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, edgecolor='k')

plt.axis('tight')
x1_min, x1_max, x2_min, x2_max = X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()

X1, X2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
Y = np.matmul(np.c_[X1.ravel(), X2.ravel()],w[:2])+w[2]

#color plot the original function
Y = Y.reshape(X1.shape)
mesh = plt.pcolormesh(X1, X2, Y, linewidth=0, rasterized=True)
con = plt.contour(X1, X2, Y, colors=['k']*10, linestyles=['--']*10, levels=np.linspace(y.min(), y.max(), num=5))
plt.clabel(con, inline=1, fontsize=10)
zeroc = plt.contour(X1, X2, Y, colors=['k'], linestyles=['-'], levels=[0])
plt.clabel(zeroc, inline=1, fontsize=10)
plt.colorbar(mesh, extendfrac='auto')

plt.savefig('custsvrcomp1.pdf', bbox_inches="tight")

#sklearn
plt.figure()

X_TrainAndValidation, X_Test, y_TrainAndValidation, y_Test = train_test_split(X, y, test_size = 1/3)

param_grid = [
  {'C': [0.5, 1, 2, 4, 8, 16], 'epsilon':[0.0001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10], 'kernel': ['linear']},
 ]

result = SVR_gridsearch_holdout(X, y, svm.SVR, param_grid, 20, 25, runs=5)
best_params = result[0]

svr_lin = svm.SVR(**best_params)
svr_lin.fit(X_Train,y_Train)
plt.title("Scikit-learn SVR, test score: %f" % svr_lin.score(X_Test,y_Test))

plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, edgecolor='k')
plt.axis('tight')

Ylsvr = svr_lin.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)
meshlsvr = plt.pcolormesh(X1, X2, Ylsvr, linewidth=0, rasterized=True)
meshlsvr.set_edgecolor('face')
conlsvr = plt.contour(X1, X2, Ylsvr, colors=['k']*10, linestyles=['--']*10, levels=np.linspace(y.min(), y.max(), num=5))
plt.clabel(conlsvr, inline=1, fontsize=10)
zeroclsvr = plt.contour(X1, X2, Ylsvr, colors=['k'], linestyles=['-'], levels=[0])
plt.clabel(zeroclsvr, inline=1, fontsize=10)
plt.colorbar(meshlsvr, extendfrac='auto')

plt.savefig('custsvrcomp2.pdf', bbox_inches="tight")

#customsvr
plt.figure()

X_TrainAndValidation, X_Test, y_TrainAndValidation, y_Test = train_test_split(X, y, test_size = 1/3)

param_grid = [
  {'C': [0.5, 1, 2, 4, 8, 16], 'epsilon':[0.0001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10], 'kernel': ['linear']},
 ]

result = SVR_gridsearch_holdout(X, y, customsvr.SVR, param_grid, 20, 25, runs=5)

best_params = result[0]

best_svr = customsvr.SVR(**best_params)
best_svr.fit(X_Train,y_Train)
plt.title("Custom SVR, test score: %f" % best_svr.score(X_Test,y_Test))

plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, edgecolor='k')
plt.axis('tight')

Ysvreg = best_svr.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)
meshsvreg = plt.pcolormesh(X1, X2, Ysvreg, linewidth=0, rasterized=True)
meshsvreg.set_edgecolor('face')
consvreg = plt.contour(X1, X2, Ysvreg, colors=['k']*10, linestyles=['--']*10, levels=np.linspace(y.min(), y.max(), num=5))
plt.clabel(consvreg, inline=1, fontsize=10)
zerocsvreg = plt.contour(X1, X2, Ysvreg, colors=['k'], linestyles=['-'], levels=[0])
plt.clabel(zerocsvreg, inline=1, fontsize=10)
plt.colorbar(meshsvreg, extendfrac='auto')

plt.savefig('custsvrcomp3.pdf', bbox_inches="tight")

#nonlinear ex
plt.figure()
plt.title("Nonlinear objective function")

npoints = 50
noise = 0.1
w = np.random.randn(5)

X = np.random.rand(npoints, 2)*2 -1
X_t = np.concatenate((X, X**2), axis=1)
y = np.matmul(X_t,w[:4])+w[4] + (np.random.randn(npoints)*noise)

X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size = 1/4)

plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, edgecolor='k')

plt.axis('tight')
x1_min, x1_max, x2_min, x2_max = X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()

X1, X2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
Y = np.matmul(np.concatenate((np.c_[X1.ravel(), X2.ravel()],np.c_[X1.ravel(), X2.ravel()]**2), axis=1),w[:4])+w[4]

#color plot the original function
Y = Y.reshape(X1.shape)
mesh = plt.pcolormesh(X1, X2, Y, linewidth=0, rasterized=True)
mesh.set_edgecolor('face')
con = plt.contour(X1, X2, Y, colors=['k']*10, linestyles=['--']*10, levels=np.linspace(y.min(), y.max(), num=5))
plt.clabel(con, inline=1, fontsize=10)
zeroc = plt.contour(X1, X2, Y, colors=['k'], linestyles=['-'], levels=[0])
plt.clabel(zeroc, inline=1, fontsize=10)
plt.colorbar(mesh, extendfrac='auto')

plt.savefig('custsvrcomp4.pdf', bbox_inches="tight")

#sklearn
plt.figure()

X_TrainAndValidation, X_Test, y_TrainAndValidation, y_Test = train_test_split(X, y, test_size = 1/3)

param_grid = [
  {'C': [0.5, 1, 2, 4, 8, 16], 'epsilon':[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10], 'kernel': ['linear']},
  {'C': [0.5, 1, 2, 4, 8, 16], 'epsilon':[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']},
  {'C': [0.5, 1, 2, 4, 8, 16], 'epsilon':[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10], 'gamma': [1, 0.1, 0.01, 0.001], 'degree': [1,2,3,4], 'coef0': [-100, -10, -1, 0, 1, 10, 100], 'kernel': ['poly']}
 ]

result = SVR_gridsearch_holdout(X, y, svm.SVR, param_grid, 20, 25, runs=5)

best_params = result[0]

best_svr = svm.SVR(**best_params)
best_svr.fit(X_Train,y_Train)
plt.title("Scikit-learn SVR, test score: %f" % best_svr.score(X_Test,y_Test))

plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, edgecolor='k')
plt.axis('tight')

Ysvreg = best_svr.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)
meshsvreg = plt.pcolormesh(X1, X2, Ysvreg, linewidth=0, rasterized=True)
meshsvreg.set_edgecolor('face')
consvreg = plt.contour(X1, X2, Ysvreg, colors=['k']*10, linestyles=['--']*10, levels=np.linspace(y.min(), y.max(), num=5))
plt.clabel(consvreg, inline=1, fontsize=10)
zerocsvreg = plt.contour(X1, X2, Ysvreg, colors=['k'], linestyles=['-'], levels=[0])
plt.clabel(zerocsvreg, inline=1, fontsize=10)
plt.colorbar(meshsvreg, extendfrac='auto')

plt.savefig('custsvrcomp5.pdf', bbox_inches="tight")

#customsvr
plt.figure()

X_TrainAndValidation, X_Test, y_TrainAndValidation, y_Test = train_test_split(X, y, test_size = 1/3)

param_grid = [
  {'C': [0.5, 1, 2, 4, 8, 16], 'epsilon':[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10], 'kernel': ['linear']},
  {'C': [0.5, 1, 2, 4, 8, 16], 'epsilon':[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']},
  {'C': [0.5, 1, 2, 4, 8, 16], 'epsilon':[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10], 'gamma': [1, 0.1, 0.01, 0.001], 'degree': [1,2,3,4], 'coef0': [0, 1, 10, 100], 'kernel': ['poly']}
 ]

result = SVR_gridsearch_holdout(X, y, customsvr.SVR, param_grid, 20, 25, runs=5)

best_params = result[0]

best_svr = customsvr.SVR(**best_params)
best_svr.fit(X_Train,y_Train)
plt.title("Custom SVR, test score: %f" % best_svr.score(X_Test,y_Test))

plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, edgecolor='k')
plt.axis('tight')

Ysvreg = best_svr.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)
meshsvreg = plt.pcolormesh(X1, X2, Ysvreg, linewidth=0, rasterized=True)
meshsvreg.set_edgecolor('face')
consvreg = plt.contour(X1, X2, Ysvreg, colors=['k']*10, linestyles=['--']*10, levels=np.linspace(y.min(), y.max(), num=5))
plt.clabel(consvreg, inline=1, fontsize=10)
zerocsvreg = plt.contour(X1, X2, Ysvreg, colors=['k'], linestyles=['-'], levels=[0])
plt.clabel(zerocsvreg, inline=1, fontsize=10)
plt.colorbar(meshsvreg, extendfrac='auto')

plt.savefig('custsvrcomp6.pdf', bbox_inches="tight")
