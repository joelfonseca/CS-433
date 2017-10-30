# CS-433 Machine Learning - Project 1

The first project developed for the Machine Learning course, was about predicting from a data set wether a particle was a Higgs Boson based on thirty features. The project was run like a Kaggle Competition.

The content of this project is composed of different parts:

- the folder `competition-data` contains all the data needed for the project. It is composed of three files:

   - `sample-submission.csv`: this shows the format in which you have to present your predictions.
   - `test.csv`: the data set used to make predictions from the model.
   - `train.csv`: the data set used for training the model.

- the folder `report` contains three items:

   - `report.tex`: the report in LaTeX format.
   - `report.pdf`: the report in PDF format.
   - `figures`: this folder contains all the figures used in the report.

- the `project1.ipynb` notebook used for the development of this project.

- several `.py` files:

   - **`costs.py`**: contains all the necessary functions to compute the loss of different models.
      - `compute_mse`: computes the cost using MSE.
      - `compute_neg_log_likelihood`: computes the cost by negative log likelihood.

   - **`cross_validation.py`**: contains all the necessary functions to perfom a cross validation.
      - `cross_validation`: performs cross validation based on given model.
      - `cross_validation_demo`: call 'cross_validation' with ranges of parameters to get the best combination.
      - `build_k_indices`: builds k indices for k-fold cross validation.

   - **`features_eng.py`**: contains features engineering functions used.
      - `build_poly_feature`: builds the polynomial expansion for a feature. 
      - `build_poly_tx`: builds the polynomial expansion for a set of features.
      - `build_inv_log_x`: creates inverse log values of features passed in argument which are positive in value.

   - **`helpers.py`**: contains help functions.
      - `batch_iter`: generates a minibatch iterator for a dataset.
      - `compute_gradient`: computes gradient for gradient descent and stochastic gradient descent.
      - `sigmoid`: sigmoid function.
      - `compute_gradient_sigmoid`: computes gradient for logistic regression.
      - `learning_by_gradient_descent`: does one step of gradient descent using logistic regression.
      - `init_w`: initializes the weight vector.
      - `accuracy`: computes the accuracy of the predictions.

   - **`implementations.py`**: contains all regression methods used for this project.
      - `least_squares_GD`: linear regression using gradient descent.
      - `least_squares_SGD`: linear regression using stochastic gradient descent.
      - `least_squares`: least squares regression using normal equations.
      - `ridge_regression`: ridge regression using normal equations.
      - `logistic_regression`: logistic regression using gradient descent.
      - `reg_logistic_regression`: regularized logistic regression using gradient descent.

   - **`plots.py`**: contains a function that plots the performance of specific model.
      - `show_ridge_performance`: shows the accuracy of the ridge regression model.
      - `show_logistic_performance`: shows the accuracy of the logistic regression model.

   - **`preprocessing.py`**: contains functions developed for preprocessing the data.
      - `standardize`: standardizes the data.
      - `standardize_predef`: standardizes the date based on specific *mean* and *std*.
      - `replace_nan_by_median`: replaces the *NaN* values with the median of the corresponding feature.
      - `get_jet_masks`: returns four masks corresponding to the jet value of 0, 1, 2 and 3 respectively.

   - **`proj1_helpers.py`**: predefined help functions for the project slightly modified.
      - `load_csv_data`: loads the data from CSV.
      - `predict_labels`: predicts label based on data matrix and weight vector compatible with Kaggle.
      - `create_csv_submission`: creates CSV file for submission.

   - **`run.py`**: contains the procedure that generates the exact CSV file submitted on Kaggle.
   
