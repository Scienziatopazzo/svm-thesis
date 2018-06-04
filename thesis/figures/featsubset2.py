#!/usr/bin/python3

import sys
sys.path.append("../../experiments")
import pandas as pd
import numpy as np
from datasets import *
import matplotlib.pyplot as plt
plt.style.use('classic')
import seaborn as sns
sns.set()

dogs = load_df_dogs_2016(dropColumns=dropNonNumeric+dropDates+dropIrrelevant+dropDead+["Vrig Tric", "FE %", "EDVI", "ESVI", "Allo sist"], NApolicy='normal', censoringPolicy='max', newFeats=True)

plt.title("Correlation heatmap")
sns.heatmap(dogs.corr(), cmap="YlGnBu")
plt.savefig('featsubset2.pdf', bbox_inches="tight")
