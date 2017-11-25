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
from parameters import BATCH_SIZE, NB_EPOCHS
from model import CNN, SimpleCNN, CompleteCNN
from utils import prediction_to_np_patched, patched_to_submission_lines, concatenate_images

PREDICTION_TEST_DIR = 'predictions_test/'
SAVED_MODEL_DIR = 'saved_models/'

CUDA = True

########## Train our model ##########

if __name__ == '__main__':
	train_loader = DataLoader(TrainingSet(), num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
	model = CompleteCNN()

	if CUDA:
		model.cuda()

	print("Training...")

	#data_size = sum(1 for _ in train_loader)

	loss = 1E9
	last_loss = 0
	step_worse = 0
	for epoch in tqdm(range(NB_EPOCHS)):
		last_loss = loss
		loss = 0

		# Storing tuples of (validation_data, validation_target) for validation part
		validation_info = []

		for i, (data, target) in tqdm(enumerate(train_loader)):
			
			if CUDA:
				data, target = Variable(data).cuda(), Variable(target).cuda()
			else:
				data, target = Variable(data), Variable(target)

			loss += model.step(data, target)
			'''
			if i >= data_size // 10:
				loss += model.step(data, target)
			else:
				validation_info.append((data, target))
			'''

		# Cross-validating: TODO
		'''
		accs = []
		for validation_data, validation_target in validation_info:
			y_pred = model.predict(validation_data)
			
			# Need to call 'prediction_to_np_patched' on every pred


			print(y_pred[0].size())


			print(validation_target.size())
			acc = accuracy_score(validation_target.data.view(-1).numpy(), y_pred.data.view(-1).numpy())
			accs.append(acc)

		print("Accuracy = %f" % torch.mean(accs))
		'''

		print("Loss at epoch %d = %f" % (epoch, loss / (i+1)))

		if last_loss < loss:
			step_worse = step_worse + 1
			if step_worse == 3:
				print("BREAK")
				break
		else:
			step_worse = 0

	print("Training done.")

	model_name = "model_CompleteCNN_{}_{}_{:.4f}".format(BATCH_SIZE, NB_EPOCHS, (loss / (i+1)))
	torch.save(model, SAVED_MODEL_DIR + model_name)

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
			prediction = model.predict(Variable(data))

		# By squeezing prediction, it becomes (608x608), and we
		# get kaggle pred which is also (608*608) but black/white by patch
		if CUDA:
			kaggle_pred = prediction_to_np_patched(prediction.cpu().squeeze())
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
