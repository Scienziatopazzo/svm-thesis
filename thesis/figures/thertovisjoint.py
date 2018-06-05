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

dogs = load_df_dogs_2016(dropColumns=dropNonNumeric+dropDates+dropIrrelevant+dropDead, NApolicy='normal', newFeats=True)


with sns.axes_style('white'):
    sns.jointplot("Therapy to visit", "Survival time", dogs, kind='reg')
    plt.savefig('thertovisjoint.pdf', bbox_inches="tight")
