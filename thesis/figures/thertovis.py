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

class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


dogs = load_df_dogs_2016(dropColumns=dropNonNumeric+dropDates+dropIrrelevant+dropDead, NApolicy='normal', censoringPolicy='max', newFeats=True)


with sns.axes_style('white'):
    g1 = sns.jointplot("Therapy to visit", "Survival time", dogs, kind='reg')

fig = plt.figure(figsize=(7,11))
gs = gridspec.GridSpec(2, 1)


SeabornFig2Grid(g1, fig, gs[0])

ax2 = plt.subplot(gs[1])
plt.title("Correlation heatmap")
g2 = sns.heatmap(dogs.corr(), cmap="YlGnBu", ax=ax2)

gs.tight_layout(fig)

plt.savefig('thertovis.pdf', bbox_inches="tight")
