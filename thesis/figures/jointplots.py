#!/usr/bin/python3

import sys
sys.path.append("../../experiments")
import pandas as pd
import numpy as np
from datasets import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use('classic')
import seaborn as sns
sns.set()
#sns.set_context('paper')

dogs = load_df_dogs_2016(NApolicy='drop', dropColumns=dropNonNumeric+dropDates+dropIrrelevant, newFeats=False)

with sns.axes_style('white'):
    i = 1
    for feat in list(dogs.columns.values):
        plt.figure()
        sns.jointplot(feat, "Survival time", dogs, kind='reg', size=4)
        plt.savefig('jointplot{}.pdf'.format(i), bbox_inches="tight")
        i += 1
