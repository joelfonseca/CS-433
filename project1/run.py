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

# Datasets loading
y_train, x_train, ids_train = load_csv_data(DATA_TRAIN, LOWER_BOUND, UPPER_BOUND)
y_test, x_test, ids_test = load_csv_data(DATA_TEST, LOWER_BOUND, UPPER_BOUND)

# Create inverse log values of features which are positive in value.
inv_log_cols = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 16, 19, 21, 23, 26]
x_train_inv_log_cols = np.log(1 / (1 + x_train[:, inv_log_cols]))
x_test_inv_log_cols = np.log(1 / (1 + x_test[:, inv_log_cols]))
x_train = np.hstack((x_train, x_train_inv_log_cols))
x_test = np.hstack((x_test, x_test_inv_log_cols))

# Masks creation
jet_train_msks = get_jet_masks(x_train)

jet_test_msks = get_jet_masks(x_test)

'''
##### USE THIS IF DOING PREPROCESSING BEFORE SPLITTING #####

# Pre-processing (to redo)
x_train = replace_nan_by_median(x_train.T).T
mean_train, std_train, tx_train = standardize(x_train.T)
x_train = tx_train.T

x_test = replace_nan_by_median(x_test.T).T
x_test = standardize_predef(x_test.T, mean_train, std_train).T

############################################################
'''

# Degrees/lambda for each split of the dataset
degrees = [10, 12, 10, 10]
lambdas = [0.0001, 0.003, 0.003, 0.001]

# Creation of vector that will contain the predictions of the different splits
y_pred_final = np.zeros(len(y_test))

# go over each split
for idx in range(len(jet_train_msks)):

    # Get split
    tx_train_selected_jet = x_train[jet_train_msks[idx]]
    tx_test_selected_jet = x_test[jet_test_msks[idx]]

    y_train_selected_jet = y_train[jet_train_msks[idx]]

    ##### USE THIS IF DOING PREPROCESSING AFTER SPLITTING #####

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

    ###########################################################

    # Build poly
    tx_train_poly_selected_jet = build_poly_tx(tx_train_selected_jet, degrees[idx])
    tx_test_poly_selected_jet = build_poly_tx(tx_test_selected_jet, degrees[idx])
    
    # Compute best method
    w_selected_jet, _ = ridge_regression(y_train_selected_jet, tx_train_poly_selected_jet, lambdas[idx])

    # Compute accuracy (only used for printing)
    acc = accuracy(y_train_selected_jet, tx_train_poly_selected_jet, w_selected_jet, LOWER_BOUND, UPPER_BOUND)
    print("Accuracy:", acc)

    # pred of split + add it to the final pred
    y_test_pred = predict_labels_kaggle(w_selected_jet, tx_test_poly_selected_jet, LOWER_BOUND, UPPER_BOUND)
    y_pred_final[jet_test_msks[idx]] = y_test_pred.flatten()


create_csv_submission(ids_test, y_pred_final, OUTPUT_PATH)

print("Output prediction done") 