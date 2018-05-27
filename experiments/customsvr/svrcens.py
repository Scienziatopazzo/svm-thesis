from .svr import *
from abc import ABC, abstractmethod
import gurobipy as gpy
import math
import numpy as np

class AbstractCensSVR(AbstractSVR):
    '''
    Abstract class for censored SVRs
    '''

    def reset(self):
        super().reset()
        self.deltas = []

    def comp(self, d, y, i, j):
        if (d[i] == 1 and d[j] == 1) or (d[i] == 1 and d[j] == 0 and y[i] <= y[j]) or (d[i] == 0 and d[j] == 1 and y[i] >= y[j]):
            return 1
        else:
            return 0

    def score(self, X, y, metric="R2"):
        '''
        Scores the prediction made on samples in X with targets y
        Can use different metrics
        '''
        if len(X) != len(y):
            raise ValueError('data and labels have different length')

        if metric == "c-index":
            return self.scoreCindex(X, y)
        else:
            return super().score(X, y, metric)

    def scoreCindex(self, X, y):
        '''
        Concordance index
        '''
        y_pred = self.predict(X)
        concordant = 0
        tot = 0
        for i in range(len(X)):
            for j in range(len(X)):
                if i != j:
                    if self.comp(X[:,-1], y, i, j) == 1:
                        tot += 1
                        if ((y_pred[i] > y_pred[j]) and (y[i] > y[j])) or ((y_pred[i] < y_pred[j]) and (y[i] < y[j])) or (y[i] == y[j]):
                            concordant += 1
        return concordant / tot



class StandardCensSVR(AbstractCensSVR):
    '''
    Support Vector Machine Regression (Standard, with censored data support for scoring appropriately)
    parameters:
    float C         Penalty parameter C. (default=1.0)
    float epsilon   Parameter specifying no-penalty epsilon-tube. (default=0.1)
    string kernel   Specifies the kernel type: 'linear', 'poly', 'rbf'. (default='rbf')
    float gamma     Kernel coefficient for 'rbf' and 'poly'. If 'auto', then gamma=1/n_features. (default='auto')
    int degree      Degree of the polynomial kernel 'poly', ignored by other kernels (default=3)
    float coef0     Independent term in polynomial kernel 'poly', ignored by other kernels (default=0.0)
    bool verbose    Enable verbose Gurobi output. (default=False)
    '''
    def __init__(self, C=1.0, epsilon=0.1, kernel='rbf', gamma='auto', degree=3, coef0=0, verbose=False):
        self.C = C
        self.epsilon = epsilon
        super().__init__(kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, verbose=verbose)

    def fit(self, X, y):
        '''
        fits the model on the data
        parameters:
        ndarray X   numpy matrix of data points (shape: n_samples x n_features)
        ndarray y   numpy array of target values (shape: n_samples x 1)
        '''
        if len(X) != len(y):
            raise ValueError('data and labels have different length')

        self.reset()

        #Separating censoring indicators (deltas) from rest of input data
        self.deltas = X[:,-1]
        X = X[:,0:-1]

        l = len(X)
        a = self.model.addVars(l, 2, lb=0.0, ub=self.C, vtype=gpy.GRB.CONTINUOUS, name='a')
        self.model.update()

        fobj_kern = -(1/2) * gpy.quicksum((a[i,0] - a[i,1])*(a[j,0] - a[j,1])*self.k(X[i],X[j]) for i in range(l) for j in range(l))
        fobj_eps = -self.epsilon * gpy.quicksum(a[i,0] + a[i,1] for i in range(l))
        fobj_y = gpy.quicksum(y[i]*(a[i,0] - a[i,1]) for i in range(l))
        self.model.setObjective(fobj_kern + fobj_eps + fobj_y, gpy.GRB.MAXIMIZE)

        constr = gpy.quicksum(a[i,0] - a[i,1] for i in range(l))
        self.model.addConstr(constr, gpy.GRB.EQUAL, 0)

        self.model.optimize()
        if self.model.Status != gpy.GRB.OPTIMAL:
            raise ValueError('optimal solution not found!')

        #Saving Support Vectors and multipliers
        for i in range(l):
            if abs(a[i,0].x - a[i,1].x) >= 1/(l**2):
                self.sv.append(X[i])
                self.sv_w.append(a[i,0].x - a[i,1].x)

        #Computing b
        maxlb = -float('inf')
        minub = float('inf')
        for i in range(l):
            bound = -self.epsilon + y[i] - self.kern_sum(X[i])
            if (a[i,0].x < self.C or a[i,1].x > 1/(l**2)) and bound > maxlb:
                maxlb = bound
            if (a[i,0].x > 1/(l**2) or a[i,1].x < self.C) and bound < minub:
                minub = bound
            if maxlb == minub:
                self.b = bound
                break
        if maxlb!=minub:
            raise ValueError('cannot compute b')

    def kern_sum(self, x):
        ks = 0
        for i in range(len(self.sv)):
            ks += self.sv_w[i] * self.k(self.sv[i], x)
        return ks

    def hypothesis_f(self, x):
        return self.kern_sum(x[:-1]) + self.b



class SVCR(AbstractCensSVR):
    '''
    SVCR for censored data
    parameters:
    float C         Penalty parameter C. (default=1.0)
    string kernel   Specifies the kernel type: 'linear', 'poly', 'rbf'. (default='rbf')
    float gamma     Kernel coefficient for 'rbf' and 'poly'. If 'auto', then gamma=1/n_features. (default='auto')
    int degree      Degree of the polynomial kernel 'poly', ignored by other kernels (default=3)
    float coef0     Independent term in polynomial kernel 'poly', ignored by other kernels (default=0.0)
    bool verbose    Enable verbose Gurobi output. (default=False)
    '''
    def __init__(self, C=1.0, kernel='rbf', gamma='auto', degree=3, coef0=0, verbose=False):
        self.C = C
        super().__init__(kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, verbose=verbose)

    def fit(self, X, y):
        '''
        fits the model on the data
        parameters:
        ndarray X   numpy matrix of data points (shape: n_samples x (n_features+1))
                    IMPORTANT: last column of X corresponds to censoring indicators (deltas)
        ndarray y   numpy array of target values (shape: n_samples x 1)
        '''
        if len(X) != len(y):
            raise ValueError('data and labels have different length')

        self.reset()

        #Separating censoring indicators (deltas) from rest of input data
        self.deltas = X[:,-1]
        X = X[:,0:-1]

        l = len(X)
        a = self.model.addVars(l, 2, lb=0.0, ub=self.C, vtype=gpy.GRB.CONTINUOUS, name='a')
        self.model.update()

        fobj_kern = -(1/2) * gpy.quicksum((a[i,0] - self.deltas[i]*a[i,1])*(a[j,0] - self.deltas[j]*a[j,1])*self.k(X[i],X[j]) for i in range(l) for j in range(l))
        fobj_y = gpy.quicksum(y[i]*(a[i,0] - self.deltas[i]*a[i,1]) for i in range(l))
        self.model.setObjective(fobj_kern + fobj_y, gpy.GRB.MAXIMIZE)

        constr = gpy.quicksum(a[i,0] - self.deltas[i]*a[i,1] for i in range(l))
        self.model.addConstr(constr, gpy.GRB.EQUAL, 0)

        self.model.optimize()
        if self.model.Status != gpy.GRB.OPTIMAL:
            raise ValueError('optimal solution not found!')

        #Saving Support Vectors and multipliers
        for i in range(l):
            if abs(a[i,0].x - self.deltas[i]*a[i,1].x) >= 1/(l**2):
                self.sv.append(X[i])
                self.sv_w.append(a[i,0].x - self.deltas[i]*a[i,1].x)

        #Computing b
        maxlb = -float('inf')
        minub = float('inf')
        for i in range(l):
            bound = y[i] - self.kern_sum(X[i])
            if (a[i,0].x < self.C or a[i,1].x > 1/(l**2)) and bound > maxlb:
                maxlb = bound
            if (a[i,0].x > 1/(l**2) or (self.deltas[i]>0 and a[i,1].x < self.C)) and bound < minub:
                minub = bound
            if maxlb == minub:
                self.b = bound
                break
        if maxlb!=minub:
            #raise ValueError('cannot compute b')
            if maxlb != -float('inf'):
                self.b = maxlb
            elif minub != float('inf'):
                self.b = minub
            else:
                self.b = 0

    def kern_sum(self, x):
        ks = 0
        for i in range(len(self.sv)):
            ks += self.sv_w[i] * self.k(self.sv[i], x)
        return ks

    def hypothesis_f(self, x):
        return self.kern_sum(x[:-1]) + self.b


class SVRC(AbstractCensSVR):
    '''
    SVRC for censored data
    parameters:
    (float,float) C     Penalty parameters Cleft and Cright, penalizing Ys left and right of the real values for non-censored data points. (Cleft<Cright) (default=(1.0,2.0))
    (float,float) CC    Penalty parameters CCleft and CCright, penalizing Ys left and right of the real values for censored data points. (Cleft>Cright) (default=(2.0,1.0))
    float epsilon       Parameter specifying no-penalty epsilon-tube for non-censored data points. (default=0.1)
    float epsilonC      Parameter specifying no-penalty epsilon-tube for censored data points. (default=0.1)
    string kernel       Specifies the kernel type: 'linear', 'poly', 'rbf'. (default='rbf')
    float gamma         Kernel coefficient for 'rbf' and 'poly'. If 'auto', then gamma=1/n_features. (default='auto')
    int degree          Degree of the polynomial kernel 'poly', ignored by other kernels (default=3)
    float coef0         Independent term in polynomial kernel 'poly', ignored by other kernels (default=0.0)
    bool verbose        Enable verbose Gurobi output. (default=False)
    '''
    def __init__(self, C=(1.0,2.0), CC=(2.0,1.0), epsilon=0.1, epsilonC=0.1, kernel='rbf', gamma='auto', degree=3, coef0=0, verbose=False):
        self.C = C
        self.CC = CC
        self.epsilon = epsilon
        self.epsilonC = epsilonC
        super().__init__(kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, verbose=verbose)

    def fit(self, X, y):
        '''
        fits the model on the data
        parameters:
        ndarray X   numpy matrix of data points (shape: n_samples x (n_features+1))
                    IMPORTANT: last column of X corresponds to censoring indicators (deltas)
        ndarray y   numpy array of target values (shape: n_samples x 1)
        '''
        if len(X) != len(y):
            raise ValueError('data and labels have different length')

        self.reset()

        #Separating censoring indicators (deltas) from rest of input data
        self.deltas = X[:,-1]
        X = X[:,0:-1]

        l = len(X)
        a = self.model.addVars(l, 2, lb=0.0, vtype=gpy.GRB.CONTINUOUS, name='a')
        self.model.update()

        fobj_kern = -(1/2) * gpy.quicksum((a[i,0] - a[i,1])*(a[j,0] - a[j,1])*self.k(X[i],X[j]) for i in range(l) for j in range(l))
        fobj_eps = - gpy.quicksum((self.deltas[i]*self.epsilon + (1 - self.deltas[i])*self.epsilonC)*(a[i,0] + a[i,1]) for i in range(l))
        fobj_y = gpy.quicksum(y[i]*(a[i,0] - a[i,1]) for i in range(l))
        self.model.setObjective(fobj_kern + fobj_eps + fobj_y, gpy.GRB.MAXIMIZE)

        constr = gpy.quicksum(a[i,0] - a[i,1] for i in range(l))
        self.model.addConstr(constr, gpy.GRB.EQUAL, 0)

        for i in range(l):
            self.model.addConstr(a[i,0], gpy.GRB.LESS_EQUAL, self.deltas[i]*self.C[0] + (1 - self.deltas[i])*self.CC[0])
            self.model.addConstr(a[i,1], gpy.GRB.LESS_EQUAL, self.deltas[i]*self.C[1] + (1 - self.deltas[i])*self.CC[1])

        self.model.optimize()
        if self.model.Status != gpy.GRB.OPTIMAL:
            raise ValueError('optimal solution not found!')

        #Saving Support Vectors and multipliers
        for i in range(l):
            if abs(a[i,0].x - a[i,1].x) >= 1/(l**2):
                self.sv.append(X[i])
                self.sv_w.append(a[i,0].x - a[i,1].x)

        #Computing b
        maxlb = -float('inf')
        minub = float('inf')
        for i in range(l):
            bound = -(self.deltas[i]*self.epsilon + (1 - self.deltas[i])*self.epsilonC) + y[i] - self.kern_sum(X[i])
            if (a[i,0].x < (self.deltas[i]*self.C[0] + (1 - self.deltas[i])*self.CC[0]) or a[i,1].x > 1/(l**2)) and bound > maxlb:
                maxlb = bound
            if (a[i,0].x > 1/(l**2) or a[i,1].x < self.deltas[i]*self.C[1] + (1 - self.deltas[i])*self.CC[1]) and bound < minub:
                minub = bound
            if maxlb == minub:
                self.b = bound
                break
        if maxlb!=minub:
            raise ValueError('cannot compute b')

    def kern_sum(self, x):
        ks = 0
        for i in range(len(self.sv)):
            ks += self.sv_w[i] * self.k(self.sv[i], x)
        return ks

    def hypothesis_f(self, x):
        return self.kern_sum(x[:-1]) + self.b


class RankSVMC(AbstractCensSVR):
    '''
    RankSVMC for censored data
    parameters:
    float C         Penalty parameter C. (default=1.0)
    string kernel   Specifies the kernel type: 'linear', 'poly', 'rbf'. (default='rbf')
    float gamma     Kernel coefficient for 'rbf' and 'poly'. If 'auto', then gamma=1/n_features. (default='auto')
    int degree      Degree of the polynomial kernel 'poly', ignored by other kernels (default=3)
    float coef0     Independent term in polynomial kernel 'poly', ignored by other kernels (default=0.0)
    bool verbose    Enable verbose Gurobi output. (default=False)
    '''
    def __init__(self, C=1.0, kernel='rbf', gamma='auto', degree=3, coef0=0, verbose=False):
        self.C = C
        super().__init__(kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, verbose=verbose)

    def reset(self):
        super().reset()
        self.sv2 = []

    def fit(self, X, y):
        '''
        fits the model on the data
        parameters:
        ndarray X   numpy matrix of data points (shape: n_samples x n_features)
        ndarray y   numpy array of target values (shape: n_samples x 1)
        '''
        if len(X) != len(y):
            raise ValueError('data and labels have different length')

        self.reset()

        #Separating censoring indicators (deltas) from rest of input data
        self.deltas = X[:,-1]
        X = X[:,0:-1]

        l = len(X)
        # Compiling a list of data indicex pairs (i,j) for which y[i]>y[j] and comp(i,j)==1
        ind = []
        for i in range(l):
            for j in range(l):
                if y[i] > y[j] and self.comp(self.deltas, y, i, j) == 1:
                    ind.append((i,j))

        a = self.model.addVars(len(ind), lb=0.0, ub=self.C, vtype=gpy.GRB.CONTINUOUS, name='a')
        self.model.update()

        fobj_kern = -(1/2) * gpy.quicksum(a[i]*a[j]*(self.k(X[ind[i][0]],X[ind[j][0]]) - self.k(X[ind[i][1]],X[ind[j][0]]) - self.k(X[ind[i][0]],X[ind[j][1]]) + self.k(X[ind[i][1]],X[ind[j][1]])) for i in range(len(ind)) for j in range(len(ind)))
        fobj_a = gpy.quicksum(a[i] for i in range(len(ind)))
        self.model.setObjective(fobj_kern + fobj_a, gpy.GRB.MAXIMIZE)

        self.model.optimize()
        if self.model.Status != gpy.GRB.OPTIMAL:
            raise ValueError('optimal solution not found!')

        #Saving Support Vectors and multipliers
        for i in range(len(ind)):
            if a[i].x >= 1/(l**2):
                self.sv.append(X[ind[i][0]])
                self.sv2.append(X[ind[i][1]])
                self.sv_w.append(a[i].x)

    def kern_sum(self, x):
        ks = 0
        for i in range(len(self.sv)):
            ks += self.sv_w[i] * (self.k(self.sv[i], x) - self.k(self.sv2[i], x))
        return ks

    def hypothesis_f(self, x):
        return self.kern_sum(x[:-1])


class SimpleRankSVMC(AbstractCensSVR):
    '''
    RankSVMC for censored data, with simplified constraints
    parameters:
    float C         Penalty parameter C. (default=1.0)
    string kernel   Specifies the kernel type: 'linear', 'poly', 'rbf'. (default='rbf')
    float gamma     Kernel coefficient for 'rbf' and 'poly'. If 'auto', then gamma=1/n_features. (default='auto')
    int degree      Degree of the polynomial kernel 'poly', ignored by other kernels (default=3)
    float coef0     Independent term in polynomial kernel 'poly', ignored by other kernels (default=0.0)
    bool verbose    Enable verbose Gurobi output. (default=False)
    '''
    def __init__(self, C=1.0, kernel='rbf', gamma='auto', degree=3, coef0=0, verbose=False):
        self.C = C
        super().__init__(kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, verbose=verbose)

    def reset(self):
        super().reset()
        self.sv2 = []

    def fit(self, X, y):
        '''
        fits the model on the data
        parameters:
        ndarray X   numpy matrix of data points (shape: n_samples x n_features)
        ndarray y   numpy array of target values (shape: n_samples x 1)
        '''
        if len(X) != len(y):
            raise ValueError('data and labels have different length')

        self.reset()

        #Separating censoring indicators (deltas) from rest of input data
        self.deltas = X[:,-1]
        X = X[:,0:-1]

        l = len(X)
        # Compiling a list of data indicex pairs (i,j) for which y[i]>y[j] and comp(i,j)==1
        # IMPORTANT FOR SIMPLIFICATION: only the j with the maximum y[j] is chosen for each i
        ind = []
        for i in range(l):
            maxy_val = -float('inf')
            maxy_j = 0
            for j in range(l):
                if y[i] > y[j] and self.comp(self.deltas, y, i, j) == 1:
                    if y[j] > maxy_val:
                        maxy_val = y[j]
                        maxy_j = j
            if maxy_val > -float('inf'):
                ind.append((i,maxy_j))

        a = self.model.addVars(len(ind), lb=0.0, ub=self.C, vtype=gpy.GRB.CONTINUOUS, name='a')
        self.model.update()

        fobj_kern = -(1/2) * gpy.quicksum(a[i]*a[j]*(self.k(X[ind[i][0]],X[ind[j][0]]) - self.k(X[ind[i][1]],X[ind[j][0]]) - self.k(X[ind[i][0]],X[ind[j][1]]) + self.k(X[ind[i][1]],X[ind[j][1]])) for i in range(len(ind)) for j in range(len(ind)))
        fobj_a = gpy.quicksum(a[i] for i in range(len(ind)))
        self.model.setObjective(fobj_kern + fobj_a, gpy.GRB.MAXIMIZE)

        self.model.optimize()
        if self.model.Status != gpy.GRB.OPTIMAL:
            raise ValueError('optimal solution not found!')

        #Saving Support Vectors and multipliers
        for i in range(len(ind)):
            if a[i].x >= 1/(l**2):
                self.sv.append(X[ind[i][0]])
                self.sv2.append(X[ind[i][1]])
                self.sv_w.append(a[i].x)

    def kern_sum(self, x):
        ks = 0
        for i in range(len(self.sv)):
            ks += self.sv_w[i] * (self.k(self.sv[i], x) - self.k(self.sv2[i], x))
        return ks

    def hypothesis_f(self, x):
        return self.kern_sum(x[:-1])

class Model1(AbstractCensSVR):
    '''
    Model 1 SVR for censored data (from Van Belle paper)
    parameters:
    float C         Penalty parameter C. (default=1.0)
    string kernel   Specifies the kernel type: 'linear', 'poly', 'rbf'. (default='rbf')
    float gamma     Kernel coefficient for 'rbf' and 'poly'. If 'auto', then gamma=1/n_features. (default='auto')
    int degree      Degree of the polynomial kernel 'poly', ignored by other kernels (default=3)
    float coef0     Independent term in polynomial kernel 'poly', ignored by other kernels (default=0.0)
    bool verbose    Enable verbose Gurobi output. (default=False)
    '''
    def __init__(self, C=1.0, kernel='rbf', gamma='auto', degree=3, coef0=0, verbose=False):
        self.C = C
        super().__init__(kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, verbose=verbose)

    def reset(self):
        super().reset()
        self.sv2 = []

    def fit(self, X, y):
        '''
        fits the model on the data
        parameters:
        ndarray X   numpy matrix of data points (shape: n_samples x n_features)
        ndarray y   numpy array of target values (shape: n_samples x 1)
        '''
        if len(X) != len(y):
            raise ValueError('data and labels have different length')

        self.reset()

        #Separating censoring indicators (deltas) from rest of input data
        self.deltas = X[:,-1]
        X = X[:,0:-1]

        l = len(X)
        # Compiling a list of data indicex pairs (i,j) for which y[i]>y[j] and comp(i,j)==1
        # IMPORTANT FOR SIMPLIFICATION: only the j with the maximum y[j] is chosen for each i
        ind = []
        for i in range(l):
            maxy_val = -float('inf')
            maxy_j = 0
            for j in range(l):
                if y[i] > y[j] and self.comp(self.deltas, y, i, j) == 1:
                    if y[j] > maxy_val:
                        maxy_val = y[j]
                        maxy_j = j
            if maxy_val > -float('inf'):
                ind.append((i,maxy_j))

        a = self.model.addVars(len(ind), lb=0.0, ub=self.C, vtype=gpy.GRB.CONTINUOUS, name='a')
        self.model.update()

        fobj_kern = -(1/2) * gpy.quicksum(a[i]*a[j]*(self.k(X[ind[i][0]],X[ind[j][0]]) - self.k(X[ind[i][1]],X[ind[j][0]]) - self.k(X[ind[i][0]],X[ind[j][1]]) + self.k(X[ind[i][1]],X[ind[j][1]])) for i in range(len(ind)) for j in range(len(ind)))
        fobj_a = gpy.quicksum(a[i]*(y[ind[i][0]] - y[ind[i][1]]) for i in range(len(ind)))
        self.model.setObjective(fobj_kern + fobj_a, gpy.GRB.MAXIMIZE)

        self.model.optimize()
        if self.model.Status != gpy.GRB.OPTIMAL:
            raise ValueError('optimal solution not found!')

        #Saving Support Vectors and multipliers
        for i in range(len(ind)):
            if a[i].x >= 1/(l**2):
                self.sv.append(X[ind[i][0]])
                self.sv2.append(X[ind[i][1]])
                self.sv_w.append(a[i].x)

    def kern_sum(self, x):
        ks = 0
        for i in range(len(self.sv)):
            ks += self.sv_w[i] * (self.k(self.sv[i], x) - self.k(self.sv2[i], x))
        return ks

    def hypothesis_f(self, x):
        return self.kern_sum(x[:-1])

class Model2(AbstractCensSVR):
    '''
    Model 2 SVR for censored data (from Van Belle paper)
    parameters:
    float C_rank    Penalty parameter C that penalizes misranking. (default=1.0)
    float C_pred    Penalty parameter C that penalizes value predictions. (default=1.0)
    string kernel   Specifies the kernel type: 'linear', 'poly', 'rbf'. (default='rbf')
    float gamma     Kernel coefficient for 'rbf' and 'poly'. If 'auto', then gamma=1/n_features. (default='auto')
    int degree      Degree of the polynomial kernel 'poly', ignored by other kernels (default=3)
    float coef0     Independent term in polynomial kernel 'poly', ignored by other kernels (default=0.0)
    bool verbose    Enable verbose Gurobi output. (default=False)
    '''
    def __init__(self, C_rank=1.0, C_pred=1.0, kernel='rbf', gamma='auto', degree=3, coef0=0, verbose=False):
        self.C_rank = C_rank
        self.C_pred = C_pred
        super().__init__(kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, verbose=verbose)

    def reset(self):
        super().reset()
        self.sv_w2 = []
        self.sv2 = []

    def fit(self, X, y):
        '''
        fits the model on the data
        parameters:
        ndarray X   numpy matrix of data points (shape: n_samples x n_features)
        ndarray y   numpy array of target values (shape: n_samples x 1)
        '''
        if len(X) != len(y):
            raise ValueError('data and labels have different length')

        self.reset()

        #Separating censoring indicators (deltas) from rest of input data
        self.deltas = X[:,-1]
        X = X[:,0:-1]

        l = len(X)
        # Compiling a list of data indicex pairs (i,j) for which y[i]>y[j] and comp(i,j)==1
        # IMPORTANT FOR SIMPLIFICATION: only the j with the maximum y[j] is chosen for each i
        ind = []
        for i in range(l):
            maxy_val = -float('inf')
            maxy_j = 0
            for j in range(l):
                if y[i] > y[j] and self.comp(self.deltas, y, i, j) == 1:
                    if y[j] > maxy_val:
                        maxy_val = y[j]
                        maxy_j = j
            if maxy_val > -float('inf'):
                ind.append((i,maxy_j))

        a = self.model.addVars(len(ind), lb=0.0, ub=self.C_rank, vtype=gpy.GRB.CONTINUOUS, name='a')
        b = self.model.addVars(len(ind), 2, lb=0.0, ub=self.C_pred, vtype=gpy.GRB.CONTINUOUS, name='b')
        self.model.update()

        fobj_a_kern = -(1/2) * gpy.quicksum(a[i]*a[j]*(self.k(X[ind[i][0]],X[ind[j][0]]) - self.k(X[ind[i][1]],X[ind[j][0]]) - self.k(X[ind[i][0]],X[ind[j][1]]) + self.k(X[ind[i][1]],X[ind[j][1]])) for i in range(len(ind)) for j in range(len(ind)))
        fobj_a = gpy.quicksum(a[i]*(y[ind[i][0]] - y[ind[i][1]]) for i in range(len(ind)))
        fobj_ab_kern = - gpy.quicksum((b[i,0] - self.deltas[i]*b[i,1])*a[j]*(self.k(X[ind[i][0]],X[ind[j][0]]) - self.k(X[ind[i][0]],X[ind[j][1]])) for i in range(len(ind)) for j in range(len(ind)))
        fobj_b_kern = -(1/2) * gpy.quicksum((b[i,0] - self.deltas[i]*b[i,1])*(b[j,0] - self.deltas[j]*b[j,1])*self.k(X[ind[i][0]],X[ind[j][0]]) for i in range(len(ind)) for j in range(len(ind)))
        fobj_b = gpy.quicksum((b[i,0] - self.deltas[i]*b[i,1])*y[ind[i][0]] for i in range(len(ind)))
        self.model.setObjective(fobj_a_kern + fobj_a + fobj_ab_kern + fobj_b_kern + fobj_b , gpy.GRB.MAXIMIZE)

        constr = gpy.quicksum(b[i,0] - self.deltas[i]*b[i,1] for i in range(len(ind)))
        self.model.addConstr(constr, gpy.GRB.EQUAL, 0)

        self.model.optimize()
        if self.model.Status != gpy.GRB.OPTIMAL:
            raise ValueError('optimal solution not found!')

        #Saving Support Vectors and multipliers
        for i in range(len(ind)):
            # weight for sv1
            if abs(a[i].x + b[i,0].x - self.deltas[i]*b[i,1].x) >= 1/(len(ind)**2):
                self.sv.append(X[ind[i][0]])
                self.sv_w.append(a[i].x + b[i,0].x - self.deltas[i]*b[i,1].x)
            # weight for sv2
            if a[i].x >= 1/(len(ind)**2):
                self.sv2.append(X[ind[i][1]])
                self.sv_w2.append(a[i].x)

        #Computing b
        maxlb = -float('inf')
        minub = float('inf')
        for i in range(len(ind)):
            bound = y[ind[i][0]] - self.kern_sum(X[ind[i][0]])
            if (b[i,0].x < self.C_pred or b[i,1].x > 1/(len(ind)**2)) and bound > maxlb:
                maxlb = bound
            if (b[i,0].x > 1/(len(ind)**2) or (self.deltas[i]>0 and b[i,1].x < self.C_pred)) and bound < minub:
                minub = bound
            if maxlb == minub:
                self.b = bound
                break
        if maxlb!=minub:
            #raise ValueError('cannot compute b')
            if maxlb != -float('inf'):
                self.b = maxlb
            elif minub != float('inf'):
                self.b = minub
            else:
                self.b = 0

    def kern_sum(self, x):
        ks = 0
        for i in range(len(self.sv)):
            ks += self.sv_w[i] * self.k(self.sv[i], x)
        for i in range(len(self.sv2)):
            ks += self.sv_w2[i] * (-self.k(self.sv2[i], x))
        return ks

    def hypothesis_f(self, x):
        return self.kern_sum(x[:-1]) + self.b
