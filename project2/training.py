#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
"""
    Training module for different models.
"""

import numpy as np
from random import shuffle
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader

import gc
import datetime
from tqdm import tqdm

from loader import TrainingSet
from utils import train_valid_split, snapshot
from parameters import BATCH_SIZES, CUDA, RATIO, SEED, LEARNING_RATES, ACTIVATION_FUNCTIONS
from model import CNN
from paths import SAVED_MODEL_DIR

RUN_TIME = '{:%Y-%m-%d_%H-%M}' .format(datetime.datetime.now())

def train():

    for batch_size in BATCH_SIZES:

        # Load the data
        train_loader = DataLoader(TrainingSet(), num_workers=4, batch_size=batch_size, shuffle=True)

        # Create training and validation split
        train_data, train_targets, valid_data, valid_targets = train_valid_split(train_loader, RATIO, SEED)

        # Combine train/validation data and targets as tuples
        train_data_and_targets = list(zip(train_data, train_targets))
        valid_data_and_targets = list(zip(valid_data, valid_targets))

        for learning_rate in LEARNING_RATES:
            for activation_function in ACTIVATION_FUNCTIONS:

                print('Training with batch size: ', batch_size, ', learning rate: ', learning_rate, ' and activation function: ', activation_function)
                
                # Create the model
                model = CNN(learning_rate, activation_function)

                if CUDA:
                    model.cuda()
                
                MODEL_NAME = 'CNN'
                RUN_NAME = MODEL_NAME + '_{}_{:.0e}_{}' .format(batch_size, learning_rate, activation_function)

                epoch = 0
                best_acc = (0,0)
                history = []
                while True:
                    
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

                    # Save the best model
                    if acc_epoch > best_acc[1]:
                        best_acc = (epoch, acc_epoch)
                        snapshot(SAVED_MODEL_DIR, RUN_TIME, RUN_NAME, True, model.state_dict())

                    # Check that the model is not doing worst over the time
                    if best_acc[0] + 10 < epoch :
                        print('Overfitting. Stopped at epoch {}.' .format(epoch))
                        break

                    epoch += 1 


if __name__ == '__main__':
    train()