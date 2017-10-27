""" Pyhton file to get the submission used on Kaggle
Authors: Fonseca Joël, Moullet Valentin, Laville Quentin
"""

# import
import numpy as np
from implementations import *

# constant definition
DATA_FOLDER = "data/"
DATA_TEST = DATA_FOLDER + "test.csv"
DATA_TRAIN = DATA_FOLDER + "train.csv"

OUTPUT_PATH = 'results.csv'

LOWER_BOUND = -1
UPPER_BOUND = 1

# datasets loading
y_train, x_train, ids_train = load_csv_data(DATA_TRAIN, LOWER_BOUND, UPPER_BOUND)
y_test, x_test, ids_test = load_csv_data(DATA_TEST, LOWER_BOUND, UPPER_BOUND)

# masks loading
jet_train_msks = get_jet_masks(x_train)
jet_test_msks = get_jet_masks(x_test)

# degrees/lambda for each split of the dataset
degrees = [0, 0, 0, 0]
lambdas = [0, 0, 0, 0]

# the predictions of the different splits will go there
y_pred_final = np.zeros(len(y_test))

# go over each split
for idx in range(0, len(jet_train_msks)):

	# get split
	tx_train_selected_jet = x_train[jet_train_msks[idx]].T
	tx_test_selected_jet = x_test[jet_test_msks[idx]].T

	y_train_selected_jet = y_train[jet_train_msks[idx]]

	# Remove columns full of NaN
    tx_train_selected_jet = tx_train_selected_jet[~np.all(np.isnan(tx_train_selected_jet), axis=1)]
    tx_test_selected_jet = tx_test_selected_jet[~np.all(np.isnan(tx_test_selected_jet), axis=1)]

    # Remove columns without standard deviation at all
    tx_train_selected_jet = tx_train_selected_jet[np.nanstd(tx_train_selected_jet, axis=1) != 0]
    tx_test_selected_jet = tx_test_selected_jet[np.nanstd(tx_test_selected_jet, axis=1) != 0]
    
    # Replace remaining NaN by median
    tx_train_selected_jet = replace_nan_by_median(tx_train_selected_jet)
    tx_test_selected_jet = replace_nan_by_median(tx_test_selected_jet)

    # Standardize features
    mean_train_selected_jet, std_train_selected_jet, tx_train_selected_jet = standardize(tx_train_selected_jet)
    tx_test_selected_jet = standardize_predef(tx_test_selected_jet, mean_train_selected_jet, std_train_selected_jet)

     # Build poly
    tx_train_poly_selected_jet = build_poly_tx(current_tx_train, degrees[i])
    tx_test_poly_selected_jet = build_poly_tx(current_tx_test, degrees[i])
    
    # Compute best method
    w_selected_jet, _ = ridge_regression(y_train_selected_jet, tx_train_poly_selected_jet, lambdas[i])

    # pred of split + add it to the final pred
    y_test_pred = predict_labels_kaggle(w_selected_jet, tx_test_poly_selected_jet.T, LOWER_BOUND, UPPER_BOUND)
    y_pred_final[jet_test_msks[i]] = y_test_pred.flatten()


create_csv_submission(ids_test, y_pred_final, OUTPUT_PATH)

print("Output prediction done") 