import torch
import numpy
from postprocessing import delete_outlier, tetris_shape_cleaner, border_cleaner, region_cleaner, naive_cleaner
from parameters import POSTPROCESSING

from parameters import THRESHOLD_ROAD 


def prediction_to_np_patched(img):
	width = int(img.size(0) / 16)
	height = int(img.size(1) / 16)

	# Should we round? Maybe it could be good to let the values
	# between 0 and 1, and then use them directly to compute the sum..?
	#new_img = torch.round(img).data.numpy()
	new_img = img.data.numpy()

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

	#print(new_img)
	if POSTPROCESSING:
		delete_outlier(new_img, 16)
		tetris_shape_cleaner(new_img, 16)
		border_cleaner(new_img, 16)
		region_cleaner(new_img, 16)
		naive_cleaner(new_img, 16)
	#print(new_img)

	# need to update
	#print("Number of roads: %d over %d patches (%.2f%%)" % (roads, width * height, roads / (width * height) * 100.0))
	return new_img


def patched_to_submission_lines(img, img_number):
	width = int(img.shape[0] / 16)
	height = int(img.shape[1] / 16)
	for w in range(width):
		for h in range(height):
			if img[h*16, w*16] == 1:
				label = 1
			else:
				label = 0

			yield ("{:03d}_{}_{},{}".format(img_number, w*16, h*16, label))


def concatenate_images(img, gt_img):
	def img_float_to_uint8(img):
		rimg = img - numpy.min(img)
		if numpy.max(rimg) == 0:
			return rimg.astype(numpy.uint8)
		else:
			return (rimg / numpy.max(rimg) * 255).round().astype(numpy.uint8)

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
