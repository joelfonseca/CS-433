import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from PIL import Image
import glob, sys, os, random
from tqdm import tqdm
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms

from parameters import IMG_PATCH_SIZE, DATA_AUGMENTATION, MAJORITY_VOTING
from preprocessing import data_augmentation
from postprocessing import add_flips

preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class TrainingSet(data.Dataset):
    def __init__(self):
        imgs = glob.glob('./data/training/images/*.png')
        labels = glob.glob('./data/training/groundtruth/*.png')
        print("*** Loading training images and groundtruth ***")

        if DATA_AUGMENTATION:
            print("*** Creating data augmentation ***")
            imgs, labels = data_augmentation(imgs, labels)
        else:
            imgs = [Image.open(img) for img in imgs]
            labels = [Image.open(label) for label in labels]

        img_patch_train = [img_crop(preprocess(img), IMG_PATCH_SIZE, IMG_PATCH_SIZE) for img in tqdm(imgs)]
        img_patch_test = [img_crop(transforms.ToTensor()(label), IMG_PATCH_SIZE, IMG_PATCH_SIZE) for label in tqdm(labels)]

        self.X = torch.cat(img_patch_train)
        self.Y = torch.cat(img_patch_test)
        
        # Need to round because groundtruth not binary (some values between 0 and 1)
        self.Y = torch.round(self.Y)

        '''
        for i, square in enumerate(self.X):
            print(square.size())
            imshow(square.permute(1, 2, 0).numpy())
            plt.show()

            print(torch.max(self.Y[i]))
            print(torch.min(self.Y[i]))
            print(self.Y[i].size())
            imshow(self.Y[i].squeeze().numpy())
            plt.show()
            break
        '''
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class TestSet(data.Dataset):
    def __init__(self):
        imgs = []
        for i in range(50):
            imgs.append('./data/test_set_images/test_%d/test_%d.png' % (i+1, i+1))
        
        for img in imgs:
            print(img)

        print("*** Loading test images ***")

        if MAJORITY_VOTING:
            imgs = add_flips(imgs)
        else:
            imgs = [Image.open(img) for img in imgs]

        self.X = torch.stack([preprocess(img) for img in tqdm(imgs)])
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], -1


# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.size(1)
    imgheight = im.size(2)
    is_2d = len(im.size()) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[:, j:j+w, i:i+h]
            list_patches.append(im_patch)

    return torch.stack(list_patches)

