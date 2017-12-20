import torch

LEARNING_RATES = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
BATCH_SIZES = [32, 64, 128]
ACTIVATION_FUNCTIONS = ['relu', 'leaky', 'prelu']

IMG_PATCH_SIZE = 80
THRESHOLD_ROAD = 64

DATA_AUGMENTATION = True
TEST_AUGMENTATION = True

SEED = 7
RATIO = 0.2

CUDA = torch.cuda.is_available()