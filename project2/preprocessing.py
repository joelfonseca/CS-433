"""Contains all the preprocessing functions used."""
from PIL import Image
import random
from parameters import SEED
import numpy as np

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

        alpha = random.randint(0,180)
        rotate_random_img_1 = img.rotate(alpha)
        rotate_random_label_1 = label.rotate(alpha)

        alpha = random.randint(0,180)
        rotate_random_img_2 = img.rotate(alpha)
        rotate_random_label_2 = label.rotate(alpha)

        alpha = random.randint(0,180)
        rotate_random_img_3 = img.rotate(alpha)
        rotate_random_label_3 = label.rotate(alpha)

        alpha = random.randint(0,180)
        rotate_random_img_4 = img.rotate(alpha)
        rotate_random_label_4 = label.rotate(alpha)

        grayscale_img = img.convert("L")
        #repeat for the 3 channels
        grayscale_img = np.stack([grayscale_img, grayscale_img, grayscale_img], 2)
        grayscale_label = label


        imgs_processed.extend([img, horizontal_flip_img, vertical_flip_img, rotate_90_img, rotate_270_img, rotate_random_img_1, rotate_random_img_2, rotate_random_img_3, rotate_random_img_4, grayscale_img])
        labels_processed.extend([label, horizontal_flip_label, vertical_flip_label, rotate_90_label, rotate_270_label, rotate_random_label_1, rotate_random_label_2, rotate_random_label_3, rotate_random_label_4, grayscale_label])

    return imgs_processed, labels_processed