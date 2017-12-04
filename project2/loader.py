from PIL import Image
import glob, sys, os, random
from tqdm import tqdm

import torch
import torch.utils.data as data
from torchvision import transforms

from parameters import IMG_PATCH_SIZE, MAJORITY_VOTING, SEED
from postprocessing import add_flips

to_PIL = transforms.ToPILImage()
from_PIL = transforms.ToTensor()
random_crop = transforms.RandomCrop(IMG_PATCH_SIZE)
preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class TrainingSet(data.Dataset):
    def __init__(self):
        imgs = glob.glob('./data/training/images/*.png')
        labels = glob.glob('./data/training/groundtruth/*.png')
        print("*** Loading training images and groundtruth ***")

        #img_patch_train = [img_crop(preprocess(Image.open(img)), IMG_PATCH_SIZE, IMG_PATCH_SIZE) for img in tqdm(imgs)]
        #img_patch_test = [img_crop(transforms.ToTensor()(Image.open(label)), IMG_PATCH_SIZE, IMG_PATCH_SIZE) for label in tqdm(labels)]

        img_patch_train = [preprocess(Image.open(img)) for img in tqdm(imgs)]        
        img_patch_test = [transforms.ToTensor()(Image.open(label)) for label in tqdm(labels)]

        self.X = torch.stack(img_patch_train)
        self.Y = torch.stack(img_patch_test)

        # Need to round because groundtruth not binary (some values between 0 and 1)
        self.Y = torch.round(self.Y)

        # Permute the data and the targets the same way
        num_patches = self.X.size(0)
        torch.manual_seed(SEED)
        idx = torch.randperm(num_patches)
        self.X = self.X[idx]
        self.Y = self.Y[idx]

        # Create validation data with 20% of data
        validation_size = num_patches//5
        self.X_validation = self.X[:validation_size]
        self.Y_validation = self.Y[:validation_size]

        # Create test data
        self.X = self.X[validation_size:]
        self.Y = self.Y[validation_size:]
    
    def __len__(self):
        return 10*len(self.X)

    def __getitem__(self, index):

        # Convert tensor to PIL image
        random_idx = random.randint(0, 79)
        img_X = to_PIL(self.X[random_idx])
        img_Y = to_PIL(self.Y[random_idx])

        # List of transformations
        functions = ['transpose', 'rotate']

        # List of arguments for each transformation
        args_transpose = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
        args_rotate = [0, random.randint(0,180)]
        args = {'transpose': random.choice(args_transpose), 'rotate': random.choice(args_rotate)}

        # Select function and make corresponding transformation
        function = random.choice(functions)
        
        img_X = getattr(img_X, function)(args[function])
        img_Y = getattr(img_Y, function)(args[function])

        # Make the same random crop for the image and the label
        seed = random.randint(0,20171201)
        random.seed(seed)
        img_X = random_crop(img_X)
        random.seed(seed)
        img_Y = random_crop(img_Y)

        #img_X.save('figures/X_' + str(random_idx) + '.png')
        #img_Y.save('figures/Y_' + str(random_idx) + '.png')

        return from_PIL(img_X), from_PIL(img_Y)

class ValidationSet(data.Dataset):
    def __init__(self, X_validation, Y_validation):
        self.X = X_validation
        self.Y = Y_validation

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
