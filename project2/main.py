import numpy
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms

from loader import TrainingSet, ValidationSet, TestSet
from parameters import LEARNING_RATE, BATCH_SIZE, IMG_PATCH_SIZE, CUDA, OUTPUT_RAW_PREDICTION, MAJORITY_VOTING
from model import CompleteCNN
from utils import prediction_to_np_patched, patched_to_submission_lines, concatenate_images, snapshot
from postprocessing import majority_voting
from plot import plot_results

from random import shuffle

import datetime

PREDICTION_TEST_DIR = 'predictions_test/'
SAVED_MODEL_DIR = 'saved_models/'
RAW_PREDICTION_DIR = 'data/raw_prediction/'
FIGURE_DIR = 'figures/'
RUN_TIME = '{:%Y-%m-%d_%H-%M}' .format(datetime.datetime.now())

'''
to load a model :

the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))
'''

SAVED_MODEL = ""
#SAVED_MODEL = SAVED_MODEL_DIR + "model_CompleteCNN_25_50_50_0.0736"

MODEL_NAME = 'CompleteCNN'
RUN_NAME = MODEL_NAME + '_{:.0e}_{}_{}' .format(LEARNING_RATE, BATCH_SIZE, IMG_PATCH_SIZE)

########## Train our model ##########

if __name__ == '__main__':

	if SAVED_MODEL == "":

		# Create train and validation sets
		train_set = TrainingSet()
		validation_set = ValidationSet(train_set.X_validation, train_set.Y_validation)

		# Create corresponding loaders
		train_loader = DataLoader(train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
		validation_loader = DataLoader(validation_set, num_workers=4, batch_size=1, shuffle=False)

		# Create model
		model = CompleteCNN()

		if CUDA:
				model.cuda()

		print("Training...")

		# Load train set
		'''train_data_and_targets = []
		validation_data_and_targets = []
		for (data, target) in train_loader:
			if CUDA:
				train_data_and_targets.append((Variable(data).cuda(), Variable(target).cuda()))
			else:
				train_data_and_targets.append((Variable(data), Variable(target)))

		# Load validation set
		for (data, target) in validation_loader:
			if CUDA:
				validation_data_and_targets.append((Variable(data).cuda(), Variable(target).cuda()))
			else:
				validation_data_and_targets.append((Variable(data), Variable(target)))'''
		
		train_loader_2 = []
		validation_loader_2 = []

		if CUDA:
			for data, target in train_loader:
				train_loader_2.append((Variable(data).cuda(), Variable(target).cuda()))
			for data, target in validation_loader:
				validation_loader_2.append((Variable(data, volatile=True).cuda(), Variable(target, volatile=True).cuda()))
		
		else:
			for data, target in train_loader:
				data, target = Variable(data), Variable(target)
			for data, target in validation_loader:
				data, target = Variable(data, volatile=True), Variable(target, volatile=True)
		
		# Define variables needed later
		loss_score_track = []
		loss_training_track = []
		# Tuple containing best (epoch, epoch_score)
		best_score = (0,0)
		loss = 0
		bar1 = tqdm(desc='EPOCHS', leave=False)
		bar1.refresh()
		# The model will keep training until it makes no progression after 10 epochs
		while True:
			epoch = bar1.n

			# Train the model
			loss_train_track = []
			for data, target in tqdm(train_loader_2, desc='BATCHES', leave=False):
				loss = model.step(data, target)
				loss_train_track.append(loss)

			# Make the validation
			loss_validation_track = []
			score_track = []
			#model.eval()
			for data, target in validation_loader_2:
				y_pred = model.predict(data)
				if CUDA:
					score = f1_score(target.cpu().data.view(-1).numpy(), y_pred.cpu().data.view(-1).numpy().round(), average='micro')
					acc = accuracy_score(target.cpu().data.view(-1).numpy(), y_pred.cpu().data.view(-1).numpy().round())
				else:
					score = f1_score(target.data.view(-1).numpy(), y_pred.data.view(-1).numpy().round(), average='micro')
					acc = accuracy_score(target.data.view(-1).numpy(), y_pred.data.view(-1).numpy().round())

				loss = F.binary_cross_entropy_with_logits(y_pred, target).data[0]
				loss_validation_track.append(loss)
				score_track.append(score)

			# Kepp track of learning process
			loss_train_epoch = numpy.mean(loss_train_track)
			loss_validation_epoch = numpy.mean(loss_validation_track)
			score_epoch = numpy.mean(score_track)
			loss_score_track.append((loss_validation_epoch, score_epoch))
			loss_training_track.append(loss_train_epoch)

			bar1.set_postfix(score=score_epoch, best_epoch = best_score[0], best_score=best_score[1], loss_validation=loss_validation_epoch, loss_train = loss_train_epoch)
			bar1.refresh()
			# Save the model if it has a better score
			if score_epoch > best_score[1]:
				best_score = (epoch, score_epoch)
				snapshot(SAVED_MODEL_DIR, RUN_TIME, RUN_NAME, model.state_dict())
				plot_results(FIGURE_DIR, RUN_TIME, RUN_NAME, loss_score_track, loss_training_track)

			# Check that the model is making progress over time
			if best_score[0] + 200 < epoch:
				print('Model is overfitting. Stopped at epoch {} with loss_train={:.5f}, loss_validation={:.5f} and score={:.5f}.' .format(epoch, loss_train_epoch, loss_validation_epoch, score_epoch))
				break

			bar1.update()
			
		print("Training done.")

		# Save results in figure until last saved model
		plot_results(FIGURE_DIR, RUN_TIME, RUN_NAME, loss_score_track, loss_training_track)

	else:

		#model = torch.load(SAVED_MODEL, map_location=lambda storage, loc: storage)
		model.load_state_dict(torch.load(SAVED_MODEL_DIR + MODEL_NAME + '.pt'))
		print("Model loaded.")



	########## Apply on test set ##########

	test_loader = DataLoader(TestSet(), num_workers=4, batch_size=1, shuffle=False)

	print("Applying model on test set and predicting...")

	# Load the best model from training
	model.load_state_dict(torch.load(SAVED_MODEL_DIR + RUN_TIME + '_' + RUN_NAME + '_best.pt'))
	model.eval()

	lines = []
	if MAJORITY_VOTING:
		# Buffer that will contain the four images needed for majority voting
		kaggle_preds = []
		# To keep track of the original image
		datas = []
		for i, (data, _) in tqdm(enumerate(test_loader)):
			# prediction is (1x1x608*608)
			if CUDA:
				prediction = model.predict(Variable(data, volatile=True).cuda())
			else: 
				prediction = model.predict(Variable(data, volatile=True))

			# By squeezing prediction, it becomes (608x608), and we
			# get kaggle pred which is also (608*608) but black/white by patch
			if CUDA:
				kaggle_pred = prediction.cpu().squeeze()
				kaggle_preds.append(kaggle_pred)
			else:
				kaggle_pred = prediction.squeeze()
				kaggle_preds.append(kaggle_pred)
			
			datas.append(data)

			# Every 4 iterations kaggle_preds will contain the four images
			# needed to do the majority voting
			if (i+1)%4 == 0:
				
				img_mv = majority_voting(kaggle_preds)
				kaggle_pred = prediction_to_np_patched(img_mv)

				# Save the prediction image (concatenated with the real image)
				concat_data = concatenate_images(datas[i-3].squeeze().permute(1, 2, 0).numpy(), kaggle_pred * 255)
				Image.fromarray(concat_data).convert('RGB').save(PREDICTION_TEST_DIR + "prediction_" + str((i+1)//4) + ".png")

				# Store the lines in the form Kaggle wants it: "{:03d}_{}_{},{}"
				for new_line in patched_to_submission_lines(kaggle_pred, ((i+1)//4)):
					lines.append(new_line)

				# Empty the buffer for the next four images
				kaggle_preds = []
	else:
		for i, (data, _) in tqdm(enumerate(test_loader)):
			# prediction is (1x1x608*608)
			if CUDA:
				prediction = model.predict(Variable(data, volatile=True).cuda())
			else: 
				prediction = model.predict(Variable(data, volatile=True))

			# By squeezing prediction, it becomes (608x608), and we
			# get kaggle pred which is also (608*608) but black/white by patch
			if CUDA:
				if OUTPUT_RAW_PREDICTION :
					kaggle_pred = prediction.cpu().squeeze().data.numpy()
				else:
					kaggle_pred = prediction_to_np_patched(prediction.cpu().squeeze())
			else:
				if OUTPUT_RAW_PREDICTION :
					kaggle_pred = prediction.squeeze().data.numpy()
				else:
					kaggle_pred = prediction_to_np_patched(prediction.squeeze())
			
			# Save the prediction image (concatenated with the real image)
			concat_data = concatenate_images(data.squeeze().permute(1, 2, 0).numpy(), kaggle_pred * 255)
			Image.fromarray(concat_data).convert('RGB').save(PREDICTION_TEST_DIR + "prediction_" + str(i+1) + ".png")

			# Store the lines in the form Kaggle wants it: "{:03d}_{}_{},{}"
			for new_line in patched_to_submission_lines(kaggle_pred, i+1):
				lines.append(new_line)
		
	# Create submission file
	with open('data/submissions/submission_pytorch.csv', 'w') as f:
	    f.write('id,prediction\n')
	    f.writelines('{}\n'.format(line) for line in lines)

	print("Predictions done.")
