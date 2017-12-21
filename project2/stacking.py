#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
"""
    Stacking all the models together to create a new one.
"""

import numpy as np
from tqdm import tqdm
import glob

from sklearn import linear_model
from sklearn.externals import joblib

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from loader import TrainingSet
from utils import create_input_regr, load_best_models
from model import CNN
from parameters import CUDA
from paths import SAVED_MODEL_DIR

def stacking():
    """Use the ensemble learning technique 'stacking' to put weights on all our models
    using a Linear regression"""

    # Load all the best models from grid search
    models = load_best_models(SAVED_MODEL_DIR)
    print('{} models loaded from grid search.' .format(len(models)))

    # Load the data
    train_loader = DataLoader(TrainingSet(stacking=True, whole=True), num_workers=4, batch_size=1, shuffle=False)

    # Initialize and build matrices for regression
    X_train = []
    y_train = []

    for i, (data, target) in enumerate(tqdm(train_loader)):
        
        if i == 0:
            
            if CUDA:
                X_train = np.r_[create_input_regr(Variable(data).cuda(), models)]
            else:
                X_train = np.r_[create_input_regr(Variable(data), models)]

            y_train = np.r_[Variable(target, volatile=True).data.view(-1).numpy()]

        else:
            
            if CUDA:
                X_train = np.r_[X_train, create_input_regr(Variable(data).cuda(), models)]
            else:
                X_train = np.r_[X_train, create_input_regr(Variable(data), models)]

            y_train = np.r_[y_train, Variable(target, volatile=True).data.view(-1).numpy()]

    
    # Create and train the regression
    regr = linear_model.LinearRegression()  
    regr.fit(X_train, y_train)

    # Save the classifier
    joblib.dump(regr, SAVED_MODEL_DIR + 'regr.pkl')

    # End message
    print('Regression model trained.')

if __name__ == '__main__':
    stacking()