#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
"""
    Comparison of different optimizers: Adam, SGD and SGD + Momentum.
"""

import numpy as np
from random import shuffle
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms

from loader import TrainingSet, TestSet
from parameters import CUDA, RATIO, SEED
from model import CNN
from utils import train_valid_split
from plot import plot_optim_acc, plot_optim_loss

OPTIMIZERS = ['Adam', 'SGD', 'SGD + Momentum']
BATCH_SIZE = 64
LEARNING_RATE = 1e-03
ACTIVATION_FUNCTION = 'leaky'
NUM_EPOCHS = 100

if __name__ == '__main__':

        # Load the data
        train_loader = DataLoader(TrainingSet(), num_workers=4, batch_size=BATCH_SIZE, shuffle=True)

        # Create training and validation split
        train_data, train_targets, valid_data, valid_targets = train_valid_split(train_loader, RATIO, SEED)

        # Combine train/validation data and targets as tuples
        train_data_and_targets = list(zip(train_data, train_targets))
        valid_data_and_targets = list(zip(valid_data, valid_targets))

        optimizer_results = []
        for optimizer in OPTIMIZERS:

            print('Training with batch size: ', BATCH_SIZE, ', learning rate: ', LEARNING_RATE, ', activation function: ', ACTIVATION_FUNCTION, ' and optimizer: ', optimizer, '.')
            
            # Create the model
            model = CNN(LEARNING_RATE, ACTIVATION_FUNCTION, optimizer)
            
            if CUDA:
                model.cuda()

            history = []
            for epoch in range(NUM_EPOCHS):
                
                # Shuffle the training data and targets in the same way
                shuffle(train_data_and_targets)

                # Train the model
                losses_training = []
                for data, target in train_data_and_targets:
                    loss = model.step(data, target)
                    losses_training.append(loss)

                # Make validation
                accs_validation = []
                for data, target in valid_data_and_targets:
                    y_pred = model.predict(data)

                    if CUDA:
                        target_numpy = target.data.view(-1).cpu().numpy()
                        pred_numpy = y_pred.data.view(-1).cpu().numpy().round()
                    else:
                        target_numpy = target.data.view(-1).numpy()
                        pred_numpy = y_pred.data.view(-1).numpy().round()

                    acc = accuracy_score(target_numpy, pred_numpy)
                    accs_validation.append(acc)
                
                # Mean of the losses of training and accuracies of validation predictions
                loss_epoch = np.mean(losses_training)
                acc_epoch = np.mean(accs_validation)
                history.append((loss_epoch, acc_epoch))
                print('Epoch: {} Training loss: {:.5f} Validation accuracy: {:.5f}' .format(epoch, loss_epoch, acc_epoch))

            # Append the optimizer results
            optimizer_results.append((optimizer, history))

        # Save the two plots
        plot_optim_acc(NUM_EPOCHS, optimizer_results)
        plot_optim_loss(NUM_EPOCHS, optimizer_results)			
