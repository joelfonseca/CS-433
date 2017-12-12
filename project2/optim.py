import numpy
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms

from loader import TrainingSet, TestSet
from parameters import CUDA, K_FOLD, SEED
from model import CNN, SimpleCNN, CompleteCNN
from utils import prediction_to_np_patched, patched_to_submission_lines, concatenate_images, train_valid_split, snapshot
from plot import plot_optim_acc, plot_optim_loss

import gc
import datetime
from random import shuffle

FIGURE_DIR = 'figures/'
RUN_TIME = '{:%Y-%m-%d_%H-%M}' .format(datetime.datetime.now())


OPTIMIZERS = ['Adam', 'SGD', 'SGD + Momentum']
BATCH_SIZE = 64
LEARNING_RATE = 1e-03
ACTIVATION_FUNCTION = 'leaky'

if __name__ == '__main__':

        train_loader = DataLoader(TrainingSet(), num_workers=4, batch_size=BATCH_SIZE, shuffle=True)

        # Create training and validation split
        train_data, train_targets, valid_data, valid_targets = train_valid_split(train_loader, K_FOLD, SEED)

        # Combine train/validation data and targets as tuples
        train_data_and_targets = list(zip(train_data, train_targets))
        valid_data_and_targets = list(zip(valid_data, valid_targets))


        histories = []
        for optimizer in OPTIMIZERS:

            print("Training with batch_size: ", BATCH_SIZE, " and learning rate: ", LEARNING_RATE, " and activation: ", ACTIVATION_FUNCTION)
            
            model = CompleteCNN(LEARNING_RATE, ACTIVATION_FUNCTION, optimizer)

            if CUDA:
                model.cuda()
            
            MODEL_NAME = 'CompleteCNN'
            RUN_NAME = MODEL_NAME + '_{}_{:.0e}_{}_{}' .format(BATCH_SIZE, LEARNING_RATE, ACTIVATION_FUNCTION, str(optimizer))

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
                
                # Mean of the losses of training and validation predictions
                loss_epoch = numpy.mean(losses_training)
                acc_epoch = numpy.mean(accs_validation)
                history.append((loss_epoch, acc_epoch))
                print("Epoch: {} Training loss: {:.5f} Validation accuracy: {:.5f}" .format(epoch, loss_epoch, acc_epoch))

            histories.append((optimizer, history))

        plot_optim_acc(histories)
        plot_optim_loss(histories)			
