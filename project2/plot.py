#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
"""
    Plot functions to create figures.
"""

import matplotlib.pyplot as plt

from paths import FIGURE_DIR

def plot_optim_acc(num_epochs, results):
    """Generates a plot with validation accuracy for different models."""

    for r in results:
        acc = [t[1] for t in r[1]]
        plt.plot(num_epochs, acc, label=r[0])

    plt.xlabel('Epoch')
    plt.title('Validation accuracy')
    leg = plt.legend(loc='lower right', shadow=True)
    leg.draw_frame(False)
    plt.savefig(FIGURE_DIR + 'optim_acc_results.png')
    plt.gcf().clear()

def plot_optim_loss(num_epochs, results):
    """Generates a plot with training loss for different models."""

    for r in results:
        loss = [t[0] for t in r[1]]
        plt.plot(num_epochs, loss, label=r[0])

    plt.xlabel('Epoch')
    plt.title('Training loss')
    leg = plt.legend(loc='upper right', shadow=True)
    leg.draw_frame(False)
    plt.savefig(FIGURE_DIR + 'optim_loss_results.png')
    plt.gcf().clear()