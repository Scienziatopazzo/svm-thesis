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

dogs = load_df_dogs_2016(NApolicy='drop', dropColumns=dropNonNumeric+dropDates+dropIrrelevant, newFeats=False)

plt.title("Correlation heatmap")
sns.heatmap(dogs.corr(), cmap="YlGnBu")
plt.savefig('heatmap.pdf', bbox_inches="tight")
