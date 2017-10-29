""" Pyhton file to get the submission used on Kaggle
Authors: Fonseca Joël, Moullet Valentin, Laville Quentin
"""

# Imports
import numpy as np

from proj1_helpers import *
from implementations import *
from helpers import *
from features_eng import *
from costs import *
from preprocessing import *

# Constants definition
DATA_FOLDER = "competition-data/"
DATA_TEST = DATA_FOLDER + "test.csv"
DATA_TRAIN = DATA_FOLDER + "train.csv"

OUTPUT_PATH = 'results.csv'

LOWER_BOUND = -1
UPPER_BOUND = 1

print('Loading data...')

# Datasets loading
y_train, x_train, ids_train = load_csv_data(DATA_TRAIN, LOWER_BOUND, UPPER_BOUND)
y_test, x_test, ids_test = load_csv_data(DATA_TEST, LOWER_BOUND, UPPER_BOUND)

print('Adding features...')

# Create inverse log values of features which are positive in value.
inv_log_cols = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 16, 19, 21, 23, 26]
x_train_inv_log_cols = np.log(1 / (1 + x_train[:, inv_log_cols]))
x_test_inv_log_cols = np.log(1 / (1 + x_test[:, inv_log_cols]))
x_train = np.hstack((x_train, x_train_inv_log_cols))
x_test = np.hstack((x_test, x_test_inv_log_cols))

# Masks creation
masks_jet_train = get_jet_masks(x_train)
masks_jet_test = get_jet_masks(x_test)

print('Going over all %d splits to create prediction...' % len(masks_jet_test))

# Degrees/lambda for each split of the dataset
degrees = [10, 12, 10, 10]
lambdas = [0.0001, 0.003, 0.003, 0.001]

# Creation of vector that will contain the predictions of the different splits
y_pred = np.zeros(len(y_test))

# Go over each split
for idx in range(len(masks_jet_train)):

    # Get split
    tx_train_selected_jet = x_train[masks_jet_train[idx]]
    tx_test_selected_jet = x_test[masks_jet_test[idx]]
    y_train_selected_jet = y_train[masks_jet_train[idx]]

    # Remove columns full of NaN
    tx_train_selected_jet = tx_train_selected_jet[:, ~np.all(np.isnan(tx_train_selected_jet), axis=0)]
    tx_test_selected_jet = tx_test_selected_jet[:, ~np.all(np.isnan(tx_test_selected_jet), axis=0)]

    # Remove columns without standard deviation at all
    tx_train_selected_jet = tx_train_selected_jet[:, np.nanstd(tx_train_selected_jet, axis=0) != 0]
    tx_test_selected_jet = tx_test_selected_jet[:, np.nanstd(tx_test_selected_jet, axis=0) != 0]
    
    # Replace remaining NaN by median
    tx_train_selected_jet = replace_nan_by_median(tx_train_selected_jet)
    tx_test_selected_jet = replace_nan_by_median(tx_test_selected_jet)

    # Standardize features
    mean_train_selected_jet, std_train_selected_jet, tx_train_selected_jet = standardize(tx_train_selected_jet)
    tx_test_selected_jet = standardize_predef(tx_test_selected_jet, mean_train_selected_jet, std_train_selected_jet)

    # Build poly
    tx_train_poly_selected_jet = build_poly_tx(tx_train_selected_jet, degrees[idx])
    tx_test_poly_selected_jet = build_poly_tx(tx_test_selected_jet, degrees[idx])
    
    # Compute best method
    w_selected_jet, _ = ridge_regression(y_train_selected_jet, tx_train_poly_selected_jet, lambdas[idx])

    # Compute accuracy (only used for printing)
    acc = accuracy(y_train_selected_jet, tx_train_poly_selected_jet, w_selected_jet, LOWER_BOUND, UPPER_BOUND)
    print("Accuracy of split %d:" % idx, acc)

    # Compute prediction of split + add it to the final pred
    y_test_pred = predict_labels(w_selected_jet, tx_test_poly_selected_jet, LOWER_BOUND, UPPER_BOUND)
    y_pred[masks_jet_test[idx]] = y_test_pred.flatten()

create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

print("Prediction created: output in %s." % OUTPUT_PATH)
