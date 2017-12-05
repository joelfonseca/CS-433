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
from utils import prediction_to_np_patched, patched_to_submission_lines, concatenate_images
from cross_validation import build_k_indices
from postprocessing import majority_voting

import gc

from random import shuffle

PREDICTION_TEST_DIR = 'predictions_test/'
SAVED_MODEL_DIR = 'saved_models/'

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
		datas = []
		targets = []

		for batch_size in BATCH_SIZES:

			del train_loader, datas, targets

			gc.collect()

			train_loader = DataLoader(TrainingSet(), num_workers=4, batch_size=batch_size, shuffle=True)

			datas = []
			targets = []

			for (data, target) in train_loader:
				datas.append(Variable(data).cuda())
				targets.append(Variable(target).cuda())

			# Create list of k indices for cross-validation
			k_indices = build_k_indices(datas, K_FOLD, SEED)


			#for k in tqdm(range(K_FOLD)):
			k = 1

			# Create the validation fold
			validation_data = [datas[i] for i in k_indices[k]]
			validation_targets = [targets[i] for i in k_indices[k]]

			# Create the training folds
			k_indices_train = numpy.delete(k_indices, k, 0)
			k_indices_train = k_indices_train.flatten()

			train_data = [datas[i] for i in k_indices_train]
			train_targets = [targets[i] for i in k_indices_train]


			# Combine train/validation data and targets as tuples
			train_data_and_targets = list(zip(train_data, train_targets))
			validation_data_and_targets = list(zip(validation_data, validation_targets))

			for learning_rate in LEARNING_RATES:
				for activation_function in ACTIVATION_FUNCTIONS:

					print("Training with batch_size: ", batch_size, " and learning rate: ", learning_rate, " and activation: ", activation_function)

					acc_epoch = 0
					last_acc_epoch = 0
					step_worse = 0

					model = CompleteCNN(learning_rate, activation_function)
					model.cuda()

					epoch = 0
					while True:
						
						# Shuffle the training data and targets in the same way
						shuffle(train_data_and_targets)
						train_data, train_targets = zip(*train_data_and_targets)

						# Train the model
						for i in range(len(train_data_and_targets)):
							#loss += model.step(train_data[i], train_targets[i])
							model.step(train_data[i], train_targets[i])

						# Make validation
						accs_validation = []
						for validation_data_, validation_target_ in validation_data_and_targets:
							y_pred = model.predict(validation_data_)
							acc = accuracy_score(validation_target_.data.view(-1).cpu().numpy(), y_pred.data.view(-1).cpu().numpy().round())
							accs_validation.append(acc)
						
						# Mean of the validation predictions
						acc_epoch = numpy.mean(accs_validation)
						print("Accuracy of fold {} at epoch {}: {:.5f}" .format(k, epoch, acc_epoch))

						# Make a save of the model every 5 epochs
						if epoch % 5 == 0:
							model_name = "model_CompleteCNN_{:.6}_{}_{}_{}_{:.5f}".format(learning_rate, batch_size, activation_function, epoch, acc_epoch)
							torch.save(model, SAVED_MODEL_DIR + model_name)

						# Check that the model is not doing worst over the time
						if last_acc_epoch > acc_epoch:
							step_worse = step_worse + 1
							if step_worse == 3:
								print("Overfitting")
								break
						else:
							step_worse = 0

						last_acc_epoch = acc_epoch

						



		'''
		else:
			
			loss = 1E9
			last_loss = 0
			step_worse = 0
			r = 0
			data_and_targets = []

			for (data, target) in train_loader:
				if CUDA:
					data_and_targets.append((Variable(data).cuda(), Variable(target).cuda()))
				else:
					data_and_targets.append((Variable(data), Variable(target)))
			

			for epoch in tqdm(range(NB_EPOCHS)):

				shuffle(data_and_targets)

				last_loss = loss
				loss = 0

				for i, (data, target) in tqdm(enumerate(data_and_targets)):
					
					loss += model.step(data, target)


				print("Loss at epoch %d = %f" % (epoch, loss / (i+1)))

				r = r + 1
				if r % 5 == 0:
					model_name = "model_CompleteCNN_{}_{}_{}_{:.4f}".format(BATCH_SIZE, NB_EPOCHS, r, (loss / (i+1)))
					torch.save(model, SAVED_MODEL_DIR + model_name)

				if last_loss < loss:
					step_worse = step_worse + 1
					if step_worse == 3:
						print("BREAK")
						break
				else:
					step_worse = 0

			print("Training done.")
			'''
		



	########## Apply on test set ##########


	if not TRAIN:
		SAVED_MODEL = ""
		#SAVED_MODEL = SAVED_MODEL_DIR + "model_CompleteCNN_25_50_50_0.0736"

		model = torch.load(SAVED_MODEL, map_location=lambda storage, loc: storage)
		print("Model loaded.")

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
