#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
"""
    Contains all the postprocessing functions developed.
"""

import numpy as np
from PIL import Image

import torch

def add_flips(imgs):
    """Creates flipped tranformations for all images passed in argument."""

    # This list will contain all the images (original + flipped tranformations)
    flipped_imgs = []

    for img in imgs:
        # For every image we create the four transformations
        img = Image.open(img)
        horizontal_flip_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        vertical_flip_img = img.transpose(Image.FLIP_TOP_BOTTOM)
        horizontal_and_vertical_flip_img = horizontal_flip_img.transpose(Image.FLIP_TOP_BOTTOM)

        # Add them to the list in a specific order
        flipped_imgs.extend([img, horizontal_flip_img, vertical_flip_img, horizontal_and_vertical_flip_img])

    return flipped_imgs

def test_augmentation_mean(imgs):
    """Merges the predictions of the same image from four transformations."""

    # Retrieve the different images in correct order
    img = imgs[0]
    horizontal_flip_img = imgs[1]
    vertical_flip_img = imgs[2]
    horizontal_and_vertical_flip_img = imgs[3]

    # Retrieve the original image for every flip
    img1 = img
    img2 = np.flip(horizontal_flip_img, 1).copy()
    img3 = np.flip(vertical_flip_img, 0).copy()
    img4 = np.flip(np.flip(horizontal_and_vertical_flip_img, 1), 0).copy()

    # Compute the mean for each pixel
    img_mean = np.mean([img1, img2, img3, img4], axis=0)
    
    return img_mean