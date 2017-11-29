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
from parameters import BATCH_SIZE, NB_EPOCHS, CUDA, K_FOLD, SEED, OUTPUT_RAW_PREDICTION, CROSS_VALIDATION, MAJORITY_VOTING
from model import CNN, SimpleCNN, CompleteCNN
from utils import prediction_to_np_patched, patched_to_submission_lines, concatenate_images
from cross_validation import build_k_indices
from postprocessing import majority_voting

from random import shuffle

PREDICTION_TEST_DIR = 'predictions_test/'
SAVED_MODEL_DIR = 'saved_models/'
RAW_PREDICTION_DIR = 'data/raw_prediction/'

'''
to load a model :

the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))
'''

SAVED_MODEL = ""
#SAVED_MODEL = SAVED_MODEL_DIR + "model_CompleteCNN_25_50_50_0.0736"

########## Train our model ##########

if __name__ == '__main__':

	if SAVED_MODEL == "":

		# Create train and validation sets
		train_set = TrainingSet()
		validation_set = ValidationSet(train_set.X_validation, train_set.Y_validation)

		# Create corresponding loaders
		train_loader = DataLoader(train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
		validation_loader = DataLoader(validation_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)

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
		
		# Define variables needed later
		loss_acc_track = []
		last_acc_epoch = 0
		bar1 = tqdm(desc='EPOCHS', leave=False)
		bar1.refresh()
		# The model will keep training until it makes no progression after 10 epochs
		while True:
			epoch = bar1.n
			# Shuffle the training data
			#shuffle(train_data_and_targets)

			# Train the model
			loss_track = []
			for data, target in tqdm(train_loader, desc='BATCHES', leave=False):
				if CUDA:
					data, target = Variable(data).cuda(), Variable(target).cuda()
				else:
					data, target = Variable(data), Variable(target)

				model.step(data, target)

			# Make the validation
			acc_track = []
			for data, target in validation_loader:
				if CUDA:
					data, target = Variable(data).cuda(), Variable(target).cuda()
				else:
					data, target = Variable(data), Variable(target)

				y_pred = model.predict(data)
				if CUDA:
					acc = f1_score(target.data.view(-1).cpu().numpy(), y_pred.data.view(-1).cpu().numpy().round(), average='micro')
				else:
					acc = f1_score(target.data.view(-1).numpy(), y_pred.data.view(-1).numpy().round(), average='micro')
				
				loss = F.binary_cross_entropy_with_logits(data, target).data[0]
				loss_track.append(loss)
				acc_track.append(acc)

			# Kepp track of learning process
			loss_epoch = numpy.mean(loss_track)
			acc_epoch = numpy.mean(acc_track)
			loss_acc_track.append((loss_epoch, acc_epoch))

			bar1.set_postfix(acc=acc_epoch, loss=loss_epoch)
			bar1.refresh()
			
			# Make a save of the model every 10 epochs
			if epoch % 10 == 0:
				model_name = "model_CompleteCNN_{}_{}_{:.5f}_{:.5f}".format(BATCH_SIZE, epoch, loss_epoch, acc_epoch)
				#torch.save(model, SAVED_MODEL_DIR + model_name)
				with open(SAVED_MODEL_DIR + model_name + '.pt', 'wb') as f:
					torch.save(model.state_dict(), f)

			# Check that the model is not doing worst over the time
			# update with epoch and best acc /!\
			if last_acc_epoch > acc_epoch:
				step_worse = step_worse + 1
				if step_worse == 10:
					print("Overfitting.")
					break
			else:
				step_worse = 0

			last_acc_epoch = acc_epoch
			
			bar1.update()
			
		print("Training done.")

	else:

		#model = torch.load(SAVED_MODEL, map_location=lambda storage, loc: storage)
		model.load_state_dict(torch.load(SAVED_MODEL_DIR + model_name + '.pt'))
		print("Model loaded.")



	########## Apply on test set ##########

	test_loader = DataLoader(TestSet(), num_workers=4, batch_size=1, shuffle=False)

	print("Applying model on test set and predicting...")

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
				prediction = model.predict(Variable(data).cuda())
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

			# Every for 4 iterations kaggle_preds will contain the four images
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
				prediction = model.predict(Variable(data).cuda())
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
