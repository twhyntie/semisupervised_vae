#!/usr/bin/env python
# -*- coding: utf-8 -*-

#...for the Operating System stuff.
import os

#...for the MATH.
import numpy as np

import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

def get_label_name(y):
    return np.where(y > 0.0)[0][0]

def justMNIST(x, save=True, name="digit", outdir="."):
    """Plot individual pixel-wise MNIST digit vector x"""
    DIM = 28
    TICK_SPACING = 4

    fig, ax = plt.subplots(1,1)
    plt.title(name)
    plt.imshow(x.reshape([DIM, DIM]), cmap="Greys",
               extent=((0, DIM) * 2), interpolation="none")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(TICK_SPACING))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(TICK_SPACING))

    # plt.show()
    if save:
        title = "mnist_{}.png".format(name)
        plt.savefig(os.path.join(outdir, title), bbox_inches="tight")

