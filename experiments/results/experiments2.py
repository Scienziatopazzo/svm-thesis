#!/usr/bin/python3

import sys
sys.path.append("..")
import pandas as pd
import numpy as np
import math
from datasets import *
from training import *
import customsvr

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.neighbors import LocalOutlierFactor

def main():
    n_reph_runs = int(sys.argv[1])
    n_overall_runs = int(sys.argv[1])

    fout = open('exp2results.txt', 'a')
    fout.write('Experiment 2: {} runs of rep. holdout, re-run {} times\n\n'.format(n_reph_runs, n_overall_runs))

    res = []
    for model in range(2):
        runrows = []
        for run in range(n_overall_runs):
            runrows.append([])
        res.append(runrows)

    for run in range(n_overall_runs):
        param_grid = [
            {'C': [1, 5, 10], 'epsilon':[0.01, 0.1, 1, 10], 'gamma': [0.01, 0.001], 'degree': [2,3], 'coef0': [0, 1, 10], 'kernel': ['poly']}
        ]

        #Removing IP Gravity, FE %, EDVI, ESVI, Allo sist. Normal NA, max censoring, Standard scaler (scaling only trainset)
        dogs = load_skl_dogs_2016(dropColumns=dropNonNumeric+dropDates+dropIrrelevant+dropDead+["IP Gravity", "FE %", "EDVI", "ESVI", "Allo sist"], NApolicy='normal', censoringPolicy='max', censSVR=True)
        X, y = dogs.data, dogs.target
        result = SVR_gridsearch_holdout(X, y, customsvr.StandardCensSVR, param_grid, 10, 15, runs=n_reph_runs, scaler=StandardScaler, censSVR=True, custom_metric="R2", other_cm=["c-index"])
        res[0][run] = [result[1], result[2][0], result[0]]
        sys.stdout.write('Computed model 0 for run {}\n'.format(run))

        #Removing IP Gravity, FS %, ESVI, Allo diast e Allo sist. Normal NA, max censoring, Standard scaler (scaling only trainset)
        dogs = load_skl_dogs_2016(dropColumns=dropNonNumeric+dropDates+dropIrrelevant+dropDead+["IP Gravity", "FS %", "ESVI", "Allo diast", "Allo sist"], NApolicy='normal', censoringPolicy='max', censSVR=True)
        X, y = dogs.data, dogs.target
        result = SVR_gridsearch_holdout(X, y, customsvr.StandardCensSVR, param_grid, 10, 15, runs=n_reph_runs, scaler=StandardScaler, censSVR=True, custom_metric="R2", other_cm=["c-index"])
        res[1][run] = [result[1], result[2][0], result[0]]
        sys.stdout.write('Computed model 1 for run {}\n'.format(run))

    for model in range(2):
        mean_score_R2 = 0
        mean_score_ci = 0
        for run in range(n_overall_runs):
            mean_score_R2 += res[model][run][0]/n_overall_runs
            mean_score_ci += res[model][run][1]/n_overall_runs
        sd_score_R2 = 0
        sd_score_ci = 0
        for run in range(n_overall_runs):
            sd_score_R2 += (res[model][run][0] - mean_score_R2)**2
            sd_score_ci += (res[model][run][1] - mean_score_ci)**2
        sd_score_R2 = math.sqrt(sd_score_R2/(n_overall_runs - 1))
        sd_score_ci = math.sqrt(sd_score_ci/(n_overall_runs - 1))

        fout.write('Model {}: mean R2 {}, sd R2 {}, mean c-index {}, sd c-index {}\nRuns:\n'.format(model, mean_score_R2, sd_score_R2, mean_score_ci, sd_score_ci))

        for run in range(n_overall_runs):
            fout.write('R2 {}, c-index {}, parameters: {}\n'.format(res[model][run][0], res[model][run][1], res[model][run][2]))

        fout.write('\n')

    fout.close()

if __name__=="__main__":
    main()
