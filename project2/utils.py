#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
"""
    Utilitary functions.
"""

import numpy

import torch
from torch.autograd import Variable

from parameters import POSTPROCESSING, THRESHOLD_ROAD, CUDA
from postprocessing import delete_outlier, tetris_shape_cleaner, border_cleaner, region_cleaner, naive_cleaner

def prediction_to_np_patched(img, tensor=True):
	"""Converts the raw predictions into patched predictions of size 16x16."""

	width = 0
	height = 0

	if tensor:
		width = int(img.size(0) / 16)
		height = int(img.size(1) / 16)
		new_img = torch.round(img).data.numpy()
	else:
		width = int(len(img) / 16)
		height = int(len(img[0]) / 16)

	# To define
	threshold = THRESHOLD_ROAD

	roads = 0
	for h in range(height):
		for w in range(width):
			road_votes = 0
			for i in range(16):
				for j in range(16):
					road_votes += new_img[16*h + i, 16*w + j]
						
			if road_votes >= threshold:
				roads += 1
				for i in range(16):
					for j in range(16):
						new_img[16*h + i, 16*w + j] = 1
			else:
				for i in range(16):
					for j in range(16):
						new_img[16*h + i, 16*w + j] = 0

						


	if POSTPROCESSING:
		delete_outlier(new_img, 16)
		tetris_shape_cleaner(new_img, 16)
		#border_cleaner(new_img, 16)
		region_cleaner(new_img, 16)
		#naive_cleaner(new_img, 16)

	return new_img


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

	rimg = img - numpy.min(img)
	if numpy.max(rimg) == 0:
		return rimg.astype(numpy.uint8)
	else:
		return (rimg / numpy.max(rimg) * 255).round().astype(numpy.uint8)

def concatenate_images(img, gt_img):
	"""Concatenates an image along with its corresponding groundthruth image."""

	nChannels = len(gt_img.shape)
	w = gt_img.shape[0]
	h = gt_img.shape[1]

	if nChannels == 3:
		cimg = numpy.concatenate((img, gt_img), axis=1)
	else:
		gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
		gt_img8 = img_float_to_uint8(gt_img)          
		gt_img_3c[:,:,0] = gt_img8
		gt_img_3c[:,:,1] = gt_img8
		gt_img_3c[:,:,2] = gt_img8
		img8 = img_float_to_uint8(img)
		cimg = numpy.concatenate((img8, gt_img_3c), axis=1)

	return cimg

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

	for (d, t) in train_loader:
		if CUDA:
			data.append(Variable(d).cuda())
			targets.append(Variable(t).cuda())
		else:
			data.append(Variable(d))
			targets.append(Variable(t))

	# Create list of k indices
	k_indices = build_k_indices(data, 1//ratio, seed)

	# Select k value
	k = 1

	# Create the validation fold
	valid_data = [data[i] for i in k_indices[k]]
	valid_targets = [targets[i] for i in k_indices[k]]

	# Create the training folds
	k_indices_train = numpy.delete(k_indices, k, 0)
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
