import gurobipy as gpy
import math
import numpy as np


def optimize_example():

    model = gpy.Model('example')
    model.setParam('OutputFlag', 0)

    x = model.addVar(name='x', lb=0, vtype=gpy.GRB.CONTINUOUS)
    y = model.addVar(name='y', lb=0, vtype=gpy.GRB.CONTINUOUS)

    model.update()

    model.setObjective(x*x + y*y, gpy.GRB.MINIMIZE)

    model.addConstr(x + y, gpy.GRB.GREATER_EQUAL, 1)

    model.optimize()

    if model.Status != gpy.GRB.OPTIMAL:
        raise ValueError('optimal solution not found!')

    print(x.x)
    print(y.x)

optimize_example()
