"""Contains all the preprocessing functions used."""
import numpy as np
from PIL import Image

def data_augmentation(imgs, labels):
    num_imgs = len(imgs)

    imgs_processed = []
    labels_processed = []

    for i in range(num_imgs):
        # same tranformations should be applied to both img and label to keep matching
        img = Image.open(imgs[i])
        label = Image.open(labels[i])

        horizontal_flip_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        horizontal_flip_label = label.transpose(Image.FLIP_LEFT_RIGHT)

        vertical_flip_img = img.transpose(Image.FLIP_TOP_BOTTOM)
        vertical_flip_label = label.transpose(Image.FLIP_TOP_BOTTOM)

        rotate_90_img = img.transpose(Image.ROTATE_90)
        rotate_90_label = label.transpose(Image.ROTATE_90)

        rotate_270_img = img.transpose(Image.ROTATE_270)
        rotate_270_label = label.transpose(Image.ROTATE_270)

        imgs_processed.extend([img, horizontal_flip_img, vertical_flip_img, rotate_90_img, rotate_270_img])
        labels_processed.extend([label, horizontal_flip_label, vertical_flip_label, rotate_90_label, rotate_270_label])

    return imgs_processed, labels_processed