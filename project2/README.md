# CS-433 Machine Learning - Project 2

The second project developed for the machine learning course was about road segmentation. A set of satellite images acquired from GoogleMaps were given, along with ground-truth images where each pixel is labeled as road or background. Our task was to train a classifier to segment roads in these images, i.e. assign a label `road=1, background=0` to each pixel.

We implemented a Convolutional Neural Network (CNN) using PyTorch as the deep learning framework.

## Project structure

The project environment setup is composed as follows:

- folder `data`: contains all the data needed for the project.

    - `test_set_images`: contains the test images.
    - `training`: contains the training images with the corresponding grountruth.

- folder `submissions`: folder where the .csv submission file will be saved.
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

## Library used

- `Python 3.6 (installed with Anaconda 5.0.1)` 
- `pytorch 0.3.0`
- `torchvision 0.2.0`
- `tqdm 4.19.5`

Anaconda comes with every other needed libs. If you don't intend to use Anaconda you might need to install other libs:
 - `matplotlib` for the plot
 - `sklearn` for the logistic regression
 - `pillow` for the images transformation
 - `numpy` for the matrix manipulation

## Instruction to run our code with GPU 

The computation to reach our submission rely heavly on GPU computation to speed up the processing. We get the time with a p2.xlarge instance on AWS, which come with a Nvidia k80 GPU.

 - train.py: ~ 85h 
 - stacking.py: ~10min
 - run.py: ~20min

 You can multiply this times by 30 if you are going to use a CPU only.

 Here we give the instruction to install the needed CUDA library, python and pytorch in order for you to run our run.py smoothly.

### Part 1: CUDA 

```sh
wget http://us.download.nvidia.com/tesla/375.51/nvidia-driver-local-repo-ubuntu1604_375.51-1_amd64.deb
sudo dpkg -i nvidia-driver-local-repo-ubuntu1604_375.51-1_amd64.deb
sudo apt-get update
sudo apt-get -y install cuda-drivers
sudo apt-get update && sudo apt-get -y upgrade
rm nvidia-driver-local-repo-ubuntu1604_375.51-1_amd64.deb
```

### Part 2: Python
```sh
wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
chmod +x Anaconda3-5.0.1-Linux-x86_64.sh
./Anaconda3-5.0.1-Linux-x86_64.sh
```

**Note that you will need to interact with the installation of Anaconda (tell 'yes', choose a path), and logout -> login at the end.**

### Part 3: Pytorch and other lib

```sh
rm Anaconda3-5.0.1-Linux-x86_64.sh
conda install -c soumith pytorch
conda install -c soumith torchvision
conda install tqdm
```

### Part 4: Add our code

Finally you need to unzip our code at the root (or wherever you want), and download the data from kaggle (https://www.kaggle.com/c/epfml17-segmentation/data). Note that you need to download the `test_set_images.zip` and `training.zip` inside the folder `data` as explained above.

### Part 5: Run

If you want to predict only, go with `python run.py`. If you want to train as well, inside `run.py` and change `USED_PRETRAINED_MODEL = False`.

### Final note

We don't advise to use Windows here. However, if you need pytorch, there is a version you can install  `conda install -c peterjc123 pytorch=0.1.12`. Note that there are bugs, especially memory leaks and you won't be able to train (predict is ok on my desktop with a Nvidia GTX 970).
 

