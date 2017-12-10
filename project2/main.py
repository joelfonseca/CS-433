import numpy
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms

from loader import TrainingSet, TestSet
from parameters import BATCH_SIZES, CUDA, K_FOLD, SEED, OUTPUT_RAW_PREDICTION, CROSS_VALIDATION, MAJORITY_VOTING, LEARNING_RATES, ACTIVATION_FUNCTIONS
from model import CNN, SimpleCNN, CompleteCNN
from utils import prediction_to_np_patched, patched_to_submission_lines, concatenate_images, train_valid_split, snapshot
from postprocessing import majority_voting
from plot import plot_results

import gc
import datetime
from random import shuffle

PREDICTION_TEST_DIR = 'predictions_test/'
SAVED_MODEL_DIR = 'saved_models/'
FIGURE_DIR = 'figures/'
RUN_TIME = '{:%Y-%m-%d_%H-%M}' .format(datetime.datetime.now())

'''
to load a model :

the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))
'''


TRAIN = True


########## Train our model ##########

if __name__ == '__main__':

	if TRAIN:

		train_loader = None

		for batch_size in BATCH_SIZES:

			del train_loader

			gc.collect()

			train_loader = DataLoader(TrainingSet(), num_workers=4, batch_size=batch_size, shuffle=True)

			# Create training and validation split
			train_data, train_targets, valid_data, valid_targets = train_valid_split(train_loader, K_FOLD, SEED)

			# Combine train/validation data and targets as tuples
			train_data_and_targets = list(zip(train_data, train_targets))
			valid_data_and_targets = list(zip(valid_data, valid_targets))

			for learning_rate in LEARNING_RATES:
				for activation_function in ACTIVATION_FUNCTIONS:

					print("Training with batch_size: ", batch_size, " and learning rate: ", learning_rate, " and activation: ", activation_function)
					
					model = CompleteCNN(learning_rate, activation_function)
					if CUDA:
						model.cuda()
					
					MODEL_NAME = 'CompleteCNN'
					RUN_NAME = MODEL_NAME + '_{}_{:.0e}_{}' .format(batch_size, learning_rate, activation_function)

					epoch = 0
					best_acc = (0,0)
					history = []
					while True:
						
						# Shuffle the training data and targets in the same way
						shuffle(train_data_and_targets)

						# Train the model
						losses_training = []
						for data, target in train_data_and_targets:
							loss = model.step(data, target)
							losses_training.append(loss)

						# Make validation
						accs_validation = []
						for data, target in valid_data_and_targets:
							y_pred = model.predict(data)

							if CUDA:
								target_numpy = target.data.view(-1).cpu().numpy()
								pred_numpy = y_pred.data.view(-1).cpu().numpy().round()
							else:
								target_numpy = target.data.view(-1).numpy()
								pred_numpy = y_pred.data.view(-1).numpy().round()

							acc = accuracy_score(target_data, pred_numpy)
							accs_validation.append(acc)
						
						# Mean of the losses of training and validation predictions
						loss_epoch = numpy.mean(losses_training)
						acc_epoch = numpy.mean(accs_validation)
						history.append((loss_epoch, acc_epoch))
						print("Epoch: {} Training loss: {:.5f} Validation accuracy: {:.5f}" .format(epoch, loss_epoch, acc_epoch))


						# Save the best model
						if acc_epoch > best_acc[1]:
							best_acc = (epoch, acc_epoch)
							snapshot(SAVED_MODEL_DIR, RUN_TIME, RUN_NAME, True, model.state_dict())
							plot_results(FIGURE_DIR, RUN_TIME, RUN_NAME, history)
						
						# Save every 5 epoch
						if epoch % 5 == 0:
							run_name_and_info = RUN_NAME + '_{:02}_{:.5f}' .format(epoch, acc_epoch)
							snapshot(SAVED_MODEL_DIR, RUN_TIME, run_name_and_info, False, model.state_dict())
							plot_results(FIGURE_DIR, RUN_TIME, RUN_NAME, history)

						# Check that the model is not doing worst over the time
						if best_acc[0] + 10 < epoch :
							print("Overfitting. Stopped at epoch {}." .format(epoch))
							break

						epoch += 1 

					plot_results(FIGURE_DIR, RUN_TIME, RUN_NAME, history)
						
	########## Apply on test set ##########


	if not TRAIN:

		test_loader = DataLoader(TestSet(), num_workers=4, batch_size=1, shuffle=False)

		SAVED_MODEL_DIR = "saved_models/"

		SAVED_MODEL_NAMES = [

			SAVED_MODEL_DIR + "2017-12-06_15-45_CompleteCNN_32_5e-03_leaky_relu_best.pt",

			SAVED_MODEL_DIR + "2017-12-06_15-45_CompleteCNN_32_5e-04_relu_best.pt",
		]

		models = []
		for model_name in SAVED_MODEL_NAMES:
			tmp = model_name.split("_")
			model = CompleteCNN(float(tmp[5]), tmp[6])
			model.load_state_dict(torch.load(model_name))
			#model =  torch.load(model_name)
			if CUDA:
				model.cuda()
			model.eval()
			models.append(model)

		
		print("Model loaded.")

		print("Applying model on test set and predicting...")

		model.eval()

		lines = []
		if MAJORITY_VOTING:
			# Buffer that will contain the four images needed for majority voting
			kaggle_preds = []
			# To keep track of the original image
			datas = []
			for i, (data, _) in tqdm(enumerate(test_loader)):

				if CUDA:
					v = Variable(data, volatile=True).cuda()
				else:
					v = Variable(data, volatile=True)

				predictions = []
				for model in models:
					# prediction is (1x1x608*608)
					prediction = model.predict(v)	

					# By squeezing prediction, it becomes (608x608), and we
					# get kaggle pred which is also (608*608) but black/white by patch
					if CUDA:
						kaggle_pred = prediction.cpu().squeeze()
					else:
						kaggle_pred = prediction.squeeze()

					predictions.append(kaggle_pred)

				# Append mean of predictions of all our models
				kaggle_preds.append(torch.mean(torch.stack(predictions), 0))
				
				datas.append(data)

				# Every for 4 iterations kaggle_preds will contain the four images
				# needed to do the majority voting
				if (i+1)%4 == 0:
					kaggle_pred = prediction_to_np_patched(majority_voting(kaggle_preds))

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
