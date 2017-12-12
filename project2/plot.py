#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
"""
    Plot functions for training loss and validation accuracy of models with different optimizers.
"""

import matplotlib.pyplot as plt

FIGURE_DIR = 'figures/'

def plot_optim_acc(histories):

    complete_name = 'optim_acc_results'
    num_epochs = range(1, len(histories[0][1])+1)

    for h in histories:
        acc = [t[1] for t in h[1]]
        plt.plot(num_epochs, acc, label=h[0])

    plt.xlabel('Epoch')
    plt.title('Validation accuracy')
    leg = plt.legend(loc='lower right', shadow=True)
    leg.draw_frame(False)
    plt.savefig(FIGURE_DIR + complete_name + '.png')
    plt.gcf().clear()

def plot_optim_loss(histories):

    complete_name = 'optim_loss_results'
    num_epochs = range(1, len(histories[0][1])+1)

    for h in histories:
        loss = [t[0] for t in h[1]]
        plt.plot(num_epochs, loss, label=h[0])

    plt.xlabel('Epoch')
    plt.title('Training loss')
    leg = plt.legend(loc='upper right', shadow=True)
    leg.draw_frame(False)
    plt.savefig(FIGURE_DIR + complete_name + '.png')
    plt.gcf().clear()