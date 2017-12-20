#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
"""
    Data loader for the PyTorch framework.
"""

import numpy as np
from PIL import Image
import glob, random
from tqdm import tqdm

import torch
import torch.utils.data as data
from torchvision import transforms

from preprocessing import data_augmentation
from postprocessing import add_flips
from utils import img_crop
from parameters import IMG_PATCH_SIZE, DATA_AUGMENTATION, TEST_AUGMENTATION

# Set of transformations
preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class TrainingSet(data.Dataset):

    def __init__(self, whole=False):

        # Load images and labels
        imgs = glob.glob('./data/training/images/*.png')
        labels = glob.glob('./data/training/groundtruth/*.png')
        print("*** Loading training images and groundtruth. ***")

        # Create data augmentation if set to True
        if DATA_AUGMENTATION:
            print("*** Creating data augmentation. ***")
            imgs, labels = data_augmentation(imgs, labels)
        else:
            imgs = [Image.open(img) for img in imgs]
            labels = [Image.open(label) for label in labels]

        # Check if we give patchs of image or the whole image 
        if whole:
            
            img_patch_data = [preprocess(img) for img in tqdm(imgs)]
            img_patch_target = [transforms.ToTensor()(label) for label in tqdm(labels)]

            self.X = torch.stack(img_patch_data)
            self.Y = torch.stack(img_patch_target)

        else:

            img_patch_data = [img_crop(preprocess(img), IMG_PATCH_SIZE, IMG_PATCH_SIZE) for img in tqdm(imgs)]
            img_patch_target = [img_crop(transforms.ToTensor()(label), IMG_PATCH_SIZE, IMG_PATCH_SIZE) for label in tqdm(labels)]

            self.X = torch.cat(img_patch_data)
            self.Y = torch.cat(img_patch_target)

        # Round target because groundtruth is not binary
        self.Y = torch.round(self.Y)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class TestSet(data.Dataset):

    def __init__(self):

        # Retrieve paths of all test images
        imgs = []
        for i in range(50):
            imgs.append('./data/test_set_images/test_%d/test_%d.png' % (i+1, i+1))

        print("*** Loading test images ***")

        # Check if we add augmentation transformations 
        if TEST_AUGMENTATION:
            imgs = add_flips(imgs)
        else:
            imgs = [Image.open(img) for img in imgs]

        self.X = torch.stack([preprocess(img) for img in tqdm(imgs)])
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], -1
