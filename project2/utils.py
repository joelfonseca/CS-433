#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
"""
    Utilitary functions.
"""

import numpy as np
import glob

import torch
from torch.autograd import Variable

from parameters import THRESHOLD_ROAD, CUDA
from model import CNN

def prediction_to_np_patched(img):
	"""Converts the raw predictions into patched predictions of size 16x16."""

	width = int(img.shape[0] / 16)
	height = int(img.shape[1] / 16)

	roads = 0
	for h in range(height):
		for w in range(width):
			road_votes = 0
			for i in range(16):
				for j in range(16):
					road_votes += img[16*h + i, 16*w + j]
						
			if road_votes >= THRESHOLD_ROAD:
				roads += 1
				for i in range(16):
					for j in range(16):
						img[16*h + i, 16*w + j] = 1
			else:
				for i in range(16):
					for j in range(16):
						img[16*h + i, 16*w + j] = 0

	return img

def patched_to_submission_lines(img, img_number):
	"""Creates the lines for the submission file for an image."""

	width = int(img.shape[0] / 16)
	height = int(img.shape[1] / 16)

	for w in range(width):
		for h in range(height):

			if img[h*16, w*16] == 1:
				label = 1
			else:
				label = 0

			yield ("{:03d}_{}_{},{}".format(img_number, w*16, h*16, label))

def img_float_to_uint8(img):
	"""Converts an image float into uint8 format."""

	rimg = img - np.min(img)
	if np.max(rimg) == 0:
		return rimg.astype(np.uint8)
	else:
		return (rimg / np.max(rimg) * 255).round().astype(np.uint8)

def concatenate_images(img, gt_img):
	"""Concatenates an image along with its corresponding groundthruth image."""

	nChannels = len(gt_img.shape)
	w = gt_img.shape[0]
	h = gt_img.shape[1]

	if nChannels == 3:
		cimg = np.concatenate((img, gt_img), axis=1)
	else:
		gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
		gt_img8 = img_float_to_uint8(gt_img)          
		gt_img_3c[:,:,0] = gt_img8
		gt_img_3c[:,:,1] = gt_img8
		gt_img_3c[:,:,2] = gt_img8
		img8 = img_float_to_uint8(img)
		cimg = np.concatenate((img8, gt_img_3c), axis=1)

	return cimg

def img_crop(im, w, h):
	"""Returns a tensor with all the patches of size wxh from the image passed in argument."""
    
	width = im.size(1)
	height = im.size(2)
	
	is_2d = len(im.size()) < 3

	list_patches = []
	for i in range(0, height, h):
		for j in range(0, width, w):
			
			if is_2d:
				im_patch = im[j:j+w, i:i+h]
			else:
				im_patch = im[:, j:j+w, i:i+h]
			
			list_patches.append(im_patch)
			
	return torch.stack(list_patches)

def build_k_indices(data_and_targets, k_fold, seed=1):
    """Builds k indices for k-fold."""

    num_row = len(data_and_targets)
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]

    return np.array(k_indices)

def train_valid_split(train_loader, ratio, seed):
	"""Splits the data into training and validation sets based on the ratio passed in argument."""

	data = []
	targets = []

	# Wrapp tensors
	for (d, t) in train_loader:
		
		if CUDA:
			data.append(Variable(d).cuda())
			targets.append(Variable(t).cuda())
		else:
			data.append(Variable(d))
			targets.append(Variable(t))

	# Create list of k indices
	k_indices = build_k_indices(data, int(1//ratio), seed)

	# Select k value (arbitrary)
	k = 1

	# Create the validation fold
	valid_data = [data[i] for i in k_indices[k]]
	valid_targets = [targets[i] for i in k_indices[k]]

	# Create the training folds
	k_indices_train = np.delete(k_indices, k, 0)
	k_indices_train = k_indices_train.flatten()

	train_data = [data[i] for i in k_indices_train]
	train_targets = [targets[i] for i in k_indices_train]

	return train_data, train_targets, valid_data, valid_targets

def snapshot(saved_model_dir, run_time, run_name, best, state_dict):
	"""Saves the model state."""
	
	# Write the full name
	if best :
		complete_name = saved_model_dir + run_time + '_' + run_name + '_best'
	else:
		complete_name = saved_model_dir + run_time + '_' + run_name
	
	# Save the model
	with open(complete_name + '.pt', 'wb') as f:
		torch.save(state_dict, f)

def create_input_regr(data, models):
	"""Creates the matrix which will be given as input for the regression."""
	
	# X will be a matrix of size NxM where N is the number of datapoints and M
	# the number of models
	X = []
	for i, model in enumerate(models):
		
		if i == 0:
			if CUDA:
				X = np.c_[model.predict(data).data.view(-1).cpu().numpy()]
			else:
				X = np.c_[model.predict(data).data.view(-1).numpy()]
		else:
			if CUDA:
				X = np.c_[X, model.predict(data).data.view(-1).cpu().numpy()]
			else:
				X = np.c_[X, model.predict(data).data.view(-1).numpy()]
	
	return X

def load_best_models(saved_model_dir):
	"""Loads all the best models from a given directory."""

	# Retrieve all the best models from the directory
	all_models = glob.glob('saved_models/*_best.pt')

	# Load and prepare them for prediction
	models = []
	for model_name in all_models:
		
		tmp = model_name.split('_')
		model = CNN(float(tmp[5]), tmp[6])

		if CUDA:
			model.load_state_dict(torch.load(model_name))
			model.cuda()
		else:
			model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
			
		model.eval()
		models.append(model)
		
	return models