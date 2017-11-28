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
from parameters import BATCH_SIZE, NB_EPOCHS, CUDA, K_FOLD, SEED
from model import CNN, SimpleCNN, CompleteCNN
from utils import prediction_to_np_patched, patched_to_submission_lines, concatenate_images
from cross_validation import build_k_indices

from random import shuffle

PREDICTION_TEST_DIR = 'predictions_test/'
SAVED_MODEL_DIR = 'saved_models/'

'''
to load a model :

the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))
'''

SAVED_MODEL = ""
#SAVED_MODEL = SAVED_MODEL_DIR + "model_CompleteCNN_25_20_20_0.1283"

########## Train our model ##########

if __name__ == '__main__':

	model = CompleteCNN()

	if SAVED_MODEL == "":
		train_loader = DataLoader(TrainingSet(), num_workers=4, batch_size=BATCH_SIZE, shuffle=True)

		if CUDA:
			model.cuda()

		print("Training...")

		step_worse = 0
		loss = 0

		data = []
		targets = []

		for (data_, target_) in train_loader:
			data.append(Variable(data_))
			targets.append(Variable(target_))

		for epoch in tqdm(range(NB_EPOCHS)):
			accs_fold = []

			for k in range(K_FOLD):
				accs = []

				# Create list of k indices for cross-validation
				k_indices = build_k_indices(data, K_FOLD, SEED)

				# Create the validation fold
				validation_data = [data[i] for i in k_indices[k]]
				validation_targets = [targets[i] for i in k_indices[k]]

				# Create the training folds
				k_indices = numpy.delete(k_indices, k, 0)
				k_indices = k_indices.flatten()

				train_data = [data[i] for i in k_indices]
				train_targets = [targets[i] for i in k_indices]

				if CUDA:
					for element in train_data+train_targets+validation_data+validation_targets:
						element.cuda()
				
				# Shuffle the training data and targets in the same way
				train_data_and_targets = list(zip(train_data, train_targets))
				shuffle(train_data_and_targets)
				train_data, train_targets = zip(*train_data_and_targets)

				# Combine validation data and targets as tuples
				validation_data_and_targets = zip(validation_data, validation_targets)

				# Train the model
				for i in range(len(train_data_and_targets)):
					loss += model.step(train_data[i], train_targets[i])

				# Make validation
				for validation_data_, validation_target_ in validation_data_and_targets:
					y_pred = model.predict(validation_data_)
					acc = accuracy_score(validation_target_.data.view(-1).numpy(), y_pred.data.view(-1).numpy().round())
					accs.append(acc)
				
				# Mean of the validation predictions
				accs_mean = numpy.mean(accs)

				# Accuracies of the k different folds
				accs_fold.append(accs_mean)

			# Accuracy of the current epoch
			acc_epoch = numpy.mean(accs_fold)
			last_acc_epoch = acc_epoch
			print("Accuracy: {:.4f}" .format(acc_epoch))

			# Make a save of the model every 5 epochs
			if epoch % 5 == 0:
				model_name = "model_CompleteCNN_{}_{}_{}_{:.4f}".format(BATCH_SIZE, NB_EPOCHS, epoch, acc_epoch)
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
			
		print("Training done.")

	else:

		model = torch.load(SAVED_MODEL, map_location=lambda storage, loc: storage)
		print("Model loaded")



	########## Apply on test set ##########

	test_loader = DataLoader(TestSet(), num_workers=4, batch_size=1, shuffle=False)

	print("Applying model on test set and predicting...")

	model.eval()

	lines = []
	for i, (data, _) in tqdm(enumerate(test_loader)):
		# prediction is (1x1x608*608)
		if CUDA:
			prediction = model.predict(Variable(data).cuda())
		else: 
			prediction = model.predict(Variable(data, volatile=True))

		# By squeezing prediction, it becomes (608x608), and we
		# get kaggle pred which is also (608*608) but black/white by patch
		if CUDA:
			kaggle_pred = prediction_to_np_patched(prediction.cpu().squeeze())
		else:
			if OUTPUT_RAW_CONTENT:
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
