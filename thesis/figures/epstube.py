#!/usr/bin/python3

import sys
sys.path.append("../../experiments")
import pandas as pd
import numpy as np
from datasets import *
import matplotlib.pyplot as plt
plt.style.use('classic')
from matplotlib import rc
rc('text', usetex=True)
rc('font', size=14)
rc('legend', fontsize=13)

np.random.seed(10)

x = np.linspace(0.0, 2, 200)
y = 0.3 * x + 1
y_eps1 = y + 0.3
y_eps2 = y - 0.3

fig, ax = plt.subplots(1)
fig.set_size_inches(5, 3.7)
ax.set_xlim([0,2])
ax.set_ylim([0,2.5])
for direction in ["right", "top"]:
    # hides borders
    ax.spines[direction].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.xticks([])
plt.yticks([])

# get width and height of axes object to compute
# matching arrowhead length and width
dps = fig.dpi_scale_trans.inverted()
bbox = ax.get_window_extent().transformed(dps)
width, height = bbox.width, bbox.height

# manual arrowhead width and length
hw = 1./20.*(2.5)
hl = 1./20.*(2)
lw = 1. # axis line width
ohg = 0.3 # arrow overhang

# compute matching arrowhead length and width
yhw = hw/(2.5)*(2)* height/width
yhl = hl/(2)*(2.5)* width/height

# draw x and y axis
ax.arrow(0, 0, 2, 0., fc='k', ec='k', lw = lw,
         head_width=hw, head_length=hl, overhang = ohg,
         length_includes_head= True, clip_on = False)

ax.arrow(0, 0, 0., 2.5, fc='k', ec='k', lw = lw,
         head_width=yhw, head_length=yhl, overhang = ohg,
         length_includes_head= True, clip_on = False)

ax.plot(x, y, color='black')
ax.fill_between(x, y_eps1, y_eps2, color='green', alpha=0.5)

X = np.random.rand(10)*2
Y = (0.3 * X + 1) + (np.random.randn(10)*0.1)
Xout = np.random.rand(4)*2
Yout = (0.3 * Xout + 1) + (np.random.randn(4)*0.3)

plt.scatter(X, Y, c='black', marker='x')
plt.scatter(Xout, Yout, c='black', marker='x')

ax.annotate("",
            xy=(1.7, 1.7*0.3 + 0.68), xycoords='data',
            xytext=(1.7, 1.7*0.3 + 1.32), textcoords='data',
            arrowprops=dict(arrowstyle="|-|"),
            )
ax.annotate("$+\\varepsilon$", fontsize=18, xy=(1.78, 1.7*0.3 + 1.31), xycoords='data')
ax.annotate("$-\\varepsilon$", fontsize=18, xy=(1.78, 1.7*0.3 + 0.61), xycoords='data')
ax.annotate("$f(x)$", fontsize=15, xy=(1.71, 1.7*0.3 + 1.05), xycoords='data')

plt.savefig('epstube.pdf', bbox_inches="tight")
