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
rc('font', family='serif')
rc('font', size=14)
rc('legend', fontsize=13)

#SVCR uncens
plt.figure()

x = np.linspace(-1.7, 1.7, 400)

def epsloss(t):
    if t<=-0.5:
        return 1.3*(-t-0.5)
    elif t>=0.5:
        return 1.3*(t-0.5)
    else:
        return 0

y = np.array([epsloss(xi) for xi in x])

fig, ax = plt.subplots(1)
fig.set_size_inches(5, 3)
ax.set_xlim([-2,2])
ax.set_ylim([-0.2,1.75])
for direction in ["right", "top", "left", "bottom"]:
    # hides borders
    ax.spines[direction].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ymax = 1.7

plt.xticks([])
plt.yticks([])

# get width and height of axes object to compute
# matching arrowhead length and width
dps = fig.dpi_scale_trans.inverted()
bbox = ax.get_window_extent().transformed(dps)
width, height = bbox.width, bbox.height

# manual arrowhead width and length
hw = 1./20.*(ymax-ymin)
hl = 1./20.*(xmax-xmin)
lw = 1. # axis line width
ohg = 0.3 # arrow overhang

# compute matching arrowhead length and width
yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

# draw x and y axis
ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw,
         head_width=hw, head_length=hl, overhang = ohg,
         length_includes_head= True, clip_on = False)

ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw,
         head_width=yhw, head_length=yhl, overhang = ohg,
         length_includes_head= True, clip_on = False)

ax.plot(x, y, color='black')

ax.annotate("$+\\varepsilon$", fontsize=18, xy=(0.35, -0.2), xycoords='data')
ax.annotate("$-\\varepsilon$", fontsize=18, xy=(-0.7, -0.2), xycoords='data')
ax.annotate("loss", fontsize=18, xy=(0, 1.73), xycoords='data', ha='center')
ax.annotate("$y - f(x)$", fontsize=14, xy=(1.6, -0.2), xycoords='data', ha='center')
ax.annotate("$C$", fontsize=14, xy=(0.96, 0.25), xycoords='data', ha='center')
ax.annotate("$C$", fontsize=14, xy=(-1, 0.25), xycoords='data', ha='center')

plt.savefig('altloss1.pdf', bbox_inches="tight")

#SVCR cens
plt.figure()

x = np.linspace(-1.7, 1.7, 400)

def epsloss(t):
    if t>=0.5:
        return 1.3*(t-0.5)
    else:
        return 0

y = np.array([epsloss(xi) for xi in x])

fig, ax = plt.subplots(1)
fig.set_size_inches(5, 3)
ax.set_xlim([-2,2])
ax.set_ylim([-0.2,1.75])
for direction in ["right", "top", "left", "bottom"]:
    # hides borders
    ax.spines[direction].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ymax = 1.7

plt.xticks([])
plt.yticks([])

# get width and height of axes object to compute
# matching arrowhead length and width
dps = fig.dpi_scale_trans.inverted()
bbox = ax.get_window_extent().transformed(dps)
width, height = bbox.width, bbox.height

# manual arrowhead width and length
hw = 1./20.*(ymax-ymin)
hl = 1./20.*(xmax-xmin)
lw = 1. # axis line width
ohg = 0.3 # arrow overhang

# compute matching arrowhead length and width
yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

# draw x and y axis
ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw,
         head_width=hw, head_length=hl, overhang = ohg,
         length_includes_head= True, clip_on = False)

ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw,
         head_width=yhw, head_length=yhl, overhang = ohg,
         length_includes_head= True, clip_on = False)

ax.plot(x, y, color='black')

ax.annotate("$+\\varepsilon$", fontsize=18, xy=(0.35, -0.2), xycoords='data')
ax.annotate("loss", fontsize=18, xy=(0, 1.73), xycoords='data', ha='center')
ax.annotate("$y - f(x)$", fontsize=14, xy=(1.6, -0.2), xycoords='data', ha='center')
ax.annotate("$C$", fontsize=14, xy=(0.96, 0.25), xycoords='data', ha='center')

plt.savefig('altloss2.pdf', bbox_inches="tight")


#SVRC uncens
plt.figure()

x = np.linspace(-1.7, 1.7, 400)

def epsloss(t):
    if t<=-0.5:
        return 1.3*(-t-0.5)
    elif t>=0.5:
        return 0.3*(t-0.5)
    else:
        return 0

y = np.array([epsloss(xi) for xi in x])

fig, ax = plt.subplots(1)
fig.set_size_inches(5, 3)
ax.set_xlim([-2,2])
ax.set_ylim([-0.2,1.75])
for direction in ["right", "top", "left", "bottom"]:
    # hides borders
    ax.spines[direction].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ymax = 1.7

plt.xticks([])
plt.yticks([])

# get width and height of axes object to compute
# matching arrowhead length and width
dps = fig.dpi_scale_trans.inverted()
bbox = ax.get_window_extent().transformed(dps)
width, height = bbox.width, bbox.height

# manual arrowhead width and length
hw = 1./20.*(ymax-ymin)
hl = 1./20.*(xmax-xmin)
lw = 1. # axis line width
ohg = 0.3 # arrow overhang

# compute matching arrowhead length and width
yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

# draw x and y axis
ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw,
         head_width=hw, head_length=hl, overhang = ohg,
         length_includes_head= True, clip_on = False)

ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw,
         head_width=yhw, head_length=yhl, overhang = ohg,
         length_includes_head= True, clip_on = False)

ax.plot(x, y, color='black')

ax.annotate("$+\\varepsilon$", fontsize=18, xy=(0.35, -0.2), xycoords='data')
ax.annotate("$-\\varepsilon$", fontsize=18, xy=(-0.7, -0.2), xycoords='data')
ax.annotate("loss", fontsize=18, xy=(0, 1.73), xycoords='data', ha='center')
ax.annotate("$y - f(x)$", fontsize=14, xy=(1.6, -0.2), xycoords='data', ha='center')
ax.annotate("$C$", fontsize=14, xy=(1.2, 0.07), xycoords='data', ha='center')
ax.annotate("$C^*$", fontsize=14, xy=(-0.8, 0.07), xycoords='data', ha='center')

plt.savefig('altloss3.pdf', bbox_inches="tight")


#SVRC cens
plt.figure()

x = np.linspace(-1.7, 1.7, 400)

def epsloss(t):
    if t<=-0.3:
        return 0.7*(-t-0.3)
    elif t>=0.3:
        return 1.3*(t-0.3)
    else:
        return 0

y = np.array([epsloss(xi) for xi in x])

fig, ax = plt.subplots(1)
fig.set_size_inches(5, 3)
ax.set_xlim([-2,2])
ax.set_ylim([-0.2,1.75])
for direction in ["right", "top", "left", "bottom"]:
    # hides borders
    ax.spines[direction].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ymax = 1.7

plt.xticks([])
plt.yticks([])

# get width and height of axes object to compute
# matching arrowhead length and width
dps = fig.dpi_scale_trans.inverted()
bbox = ax.get_window_extent().transformed(dps)
width, height = bbox.width, bbox.height

# manual arrowhead width and length
hw = 1./20.*(ymax-ymin)
hl = 1./20.*(xmax-xmin)
lw = 1. # axis line width
ohg = 0.3 # arrow overhang

# compute matching arrowhead length and width
yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

# draw x and y axis
ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw,
         head_width=hw, head_length=hl, overhang = ohg,
         length_includes_head= True, clip_on = False)

ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw,
         head_width=yhw, head_length=yhl, overhang = ohg,
         length_includes_head= True, clip_on = False)

ax.plot(x, y, color='black')

ax.annotate("$+\\varepsilon_c$", fontsize=18, xy=(0.15, -0.2), xycoords='data')
ax.annotate("$-\\varepsilon_c$", fontsize=18, xy=(-0.5, -0.2), xycoords='data')
ax.annotate("loss", fontsize=18, xy=(0, 1.73), xycoords='data', ha='center')
ax.annotate("$y - f(x)$", fontsize=14, xy=(1.6, -0.2), xycoords='data', ha='center')
ax.annotate("$C_c$", fontsize=14, xy=(0.7, 0.2), xycoords='data', ha='center')
ax.annotate("$C_c^*$", fontsize=14, xy=(-1, 0.2), xycoords='data', ha='center')

plt.savefig('altloss4.pdf', bbox_inches="tight")
