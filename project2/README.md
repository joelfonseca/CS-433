# CS-433 Machine Learning - Project 2

The second project developed for the machine learning course was about road segmentation. A set of satellite images acquired from GoogleMaps were given, along with ground-truth images where each pixel is labeled as road or background. Our task was to train a classifier to segment roads in these images, i.e. assign a label `road=1, background=0` to each pixel.

We implemented a Convolutional Neural Network (CNN) using PyTorch as the deep learning framework.

The project environment setup is composed as follows:

- folder `data`: contains all the data needed for the project.

    - ` raw_predictions`: folder where the predictions pixel by pixel will be saved.
    - `submissions`: folder where the .csv submission file will be saved.
    - `test_predictions`: folder where the test predictions will be saved.
    - `test_set_images`: contains the test images.
    - `training`: contains the training images with the corresponding grountruth.
    - `mask_to_submission.py`: converts 16x16 image to .csv submission file.
    - `submission_to_mask.py`: convert .csv submission file to 16x16 image.

- folder `figures`: contains the figures used for the report.
- folder `predictions_test`: contains the predictions (original image + 16x16 prediction).
- folder `saved_models`: contains all the models developed along this project.
- several `python modules`:

    - **`loader.py`**: contains the data loader for the PyTorch framework.

    - **`model.py`**: contains the implementation of the CNN.

    - **`optim.py`**: compares the performance of three different optimizers (Adam, SGD, SGD + Momentum).

    - **`parameters.py`**: contains several parameters needed for the environment setup.

    - **`paths.py`**: contains several paths to directories needed for the environment setup.

    - **`plot.py`**: contains the functions to plot the performance of some results.

    - **`postprocessing.py`**: contains functions developed for postprocessing the predictions.

    - **`preprocessing.py`**: contains functions developed for preprocessing the data.

    - **`run.py`**: generates the exact .csv file submitted on Kaggle.

    - **`stacking.py`**: stacks all the models available to construct a new one.

    - **`training.py`**: trains the CNN for different parameters. 

    - **`utils.py`**: contains auxiliary functions.