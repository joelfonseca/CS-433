#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
"""
    Module to generate the same submission file used for the Kaggle competition.
"""

import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob

from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn import linear_model

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from loader import TrainingSet, TestSet
from utils import prediction_to_np_patched, patched_to_submission_lines, concatenate_images, create_input_regr, load_best_models
from postprocessing import test_augmentation_mean
from training import train
from stacking import stacking
from parameters import CUDA
from model import CNN
from paths import SAVED_MODEL_DIR, PREDICTION_TEST_DIR, SUBMISSION_DIR

OUTPUT_PREDICTION_IMAGES = False


if __name__ == '__main__':

    # Check if "--train" flag is here to know if we want to train or just use the predefined models
    train = len(sys.argv) > 1 and sys.argv[1] == "--train"

    if train:
        train()
        stacking()

    # Load all the best models
    models = load_best_models(SAVED_MODEL_DIR)

    # Load the regression model
    regr = joblib.load(SAVED_MODEL_DIR + 'regr.pkl')

    # Load test set
    test_loader = DataLoader(TestSet(), num_workers=4, batch_size=1, shuffle=False)
    
    X_test = []
    y_preds = []
    flips = [] 
    lines = []
    for i, (data, _) in enumerate(tqdm(test_loader)):
        
        # Create predictions
        if CUDA:
            X_test = create_input_regr(Variable(data, volatile=True).cuda(), models)
        else:
            X_test = create_input_regr(Variable(data, volatile=True), models)
            
        y_pred = regr.predict(X_test).reshape((608, 608))

        # Store prediction along with respective flip version
        y_preds.append(y_pred)
        flips.append(data)

        # Wait until we have the four transformations of a single image before processing
        if (i+1)%4 == 0:
            
            # create Kaggle prediction (16x16)
            kaggle_pred = prediction_to_np_patched(test_augmentation_mean(y_preds))
            
            if OUTPUT_PREDICTION_IMAGES:
                # Save the prediction image (concatenated with the real image) for monitoring
                concat_data = concatenate_images(flips[i-3].squeeze().permute(1, 2, 0).numpy(), kaggle_pred * 255)
                Image.fromarray(concat_data).convert('RGB').save(PREDICTION_TEST_DIR + 'prediction_' + str((i+1)//4) + '.png')

            # Store the lines in the form Kaggle wants it: "{:03d}_{}_{},{}"
            for new_line in patched_to_submission_lines(kaggle_pred, ((i+1)//4)):
                lines.append(new_line)
            
            # Empty the buffer for the next four transformations
            y_preds = []
            
    # Create submission file
    submission_filename = SUBMISSION_DIR + 'submission_final.csv'
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        f.writelines('{}\n'.format(line) for line in lines)

    # End message
    print('Predictions done.')
    print('Output in: ' + submission_filename)
