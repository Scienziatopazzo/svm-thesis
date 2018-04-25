from abc import ABC, abstractmethod
import gurobipy as gpy
import math
import numpy as np

class ScoreMetricError(ValueError):
    pass

class AbstractSVR(ABC):
    '''
    Abstract class for Support Vector Machine Regression
    parameters:
    string kernel   Specifies the kernel type: 'linear', 'poly', 'rbf'. (default='rbf')
    float gamma     Kernel coefficient for 'rbf' and 'poly'. If 'auto', then gamma=1/n_features. (default='auto')
    int degree      Degree of the polynomial kernel 'poly', ignored by other kernels (default=3)
    float coef0     Independent term in polynomial kernel 'poly', ignored by other kernels (default=0.0)
    bool verbose    Enable verbose Gurobi output. (default=False)
    '''
    def __init__(self, kernel='rbf', gamma='auto', degree=3, coef0=0, verbose=False):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.verbose = verbose

        self.model = gpy.Model('SVR')
        self.model.setParam('OutputFlag', self.verbose)

        super().__init__()

    def reset(self):
        self.model.reset()
        self.model.remove(self.model.getVars())
        self.model.remove(self.model.getConstrs())
        self.model.update()
        self.sv = []
        self.sv_w = []
        self.b = 0

    def k(self, a, b):
        if self.kernel == 'linear':
            return np.dot(a, b)
        elif self.kernel == 'poly':
            g = self.gamma if self.gamma!='auto' else 1/len(a)
            return (g*np.dot(a, b) + self.coef0)**self.degree
        elif self.kernel == 'rbf':
            g = self.gamma if self.gamma!='auto' else 1/len(a)
            return math.exp(-g*(np.linalg.norm(a - b)**2))

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def hypothesis_f(self, x):
        pass

    def predict(self, X):
        '''
        Performs regression on samples in X
        '''
        return np.fromiter((self.hypothesis_f(x) for x in X), X.dtype, len(X))

    def score(self, X, y, metric="R2"):
        '''
        Scores the prediction made on samples in X with targets y
        Can use different metrics
        '''
        if len(X) != len(y):
            raise ValueError('data and labels have different length')

        if metric == "R2":
            return self.scoreR2(X, y)
        else:
            raise ScoreMetricError('Unknown metric selected')

    def scoreR2(self, X, y):
        '''
        Coefficient of determination R^2 = (1 - u/v)
        u = sum((y_true - y_pred) ** 2)
        v = sum((y_true - mean(y_true)) ** 2)
        Best score is 1.0, score can be negative
        '''
        y_pred = self.predict(X)
        u = np.sum((y - y_pred) ** 2)
        v = np.sum((y - np.mean(y)) ** 2)
        return 1 - (u/v)


class SVR(AbstractSVR):
    '''
    Support Vector Machine Regression (Standard)
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
        return self.kern_sum(x) + self.b
