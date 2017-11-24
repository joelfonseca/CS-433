import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import accuracy_score

from loader import TrainingSet, TestSet
from parameters import BATCH_SIZE, NB_EPOCHS
from model import CNN, DummyCNN

from functools import reduce
import itertools

# NOT FINISHED
def prediction_to_np_patched(img):
	width = int(img.size(0) / 16)
	height = int(img.size(1) / 16)

	print("max:", torch.max(img.data))
	print("min:", torch.min(img.data))

	new_img = img.data.numpy()
	print(new_img.shape)

	# To define
	threshold = 0

	for h in range(height):
		for w in range(width):
			road_votes = 0
			for i in range(16):
				for j in range(16):
					road_votes += new_img[16*h + i, 16*w + j]

			print('road_votes:', road_votes)
						
			if road_votes >= threshold:
				for i in range(16):
					for j in range(16):
						new_img[16*h + i, 16*w + j] = 1
			else:
				for i in range(16):
					for j in range(16):
						new_img[16*h + i, 16*w + j] = 0

	return new_img

# NOT FINISHED
def patched_to_submission_lines(img, img_number):
	width = int(img.shape[0] / 16)
	height = int(img.shape[1] / 16)
	for h in range(height):
		for w in range(width):
			if img[h, w] == 1:
				label = 1
			else:
				label = 0

			yield ("{:03d}_{}_{},{}".format(img_number, w, h, label))
			

########## Train our model ##########

train_loader = DataLoader(TrainingSet(), num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
model = CNN()

print("Training...")

data_size = sum(1 for _ in train_loader)
print("Data size = %d" % data_size)

for epoch in tqdm(range(NB_EPOCHS)):
	loss = 0

	# Storing tuples of (validation_data, validation_target) for validation part
	validation_info = []

	for i, (data, target) in tqdm(enumerate(train_loader)):
		data, target = Variable(data), Variable(target)
		if i >= data_size // 10:
			loss += model.step(data, target)
		else:
			validation_info.append((data, target))

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

print("Training done.")

########## Apply on test set ##########

test_loader = DataLoader(TestSet(), num_workers=4, batch_size=1, shuffle=False)

print("Applying model on test set and predicting...")

lines = []
for i, (data, _) in tqdm(enumerate(test_loader)):
	# prediction is (1x1x608*608)
	prediction = model.predict(Variable(data))

	# By squeezing prediction, it becomes (608x608), and we
	# get kaggle pred which is also (608*608) but black/white by patch
	kaggle_pred = prediction_to_np_patched(prediction.squeeze())
	imshow(kaggle_pred, cmap='gray', vmin=0, vmax=1)
	plt.show()

	# Store the lines in the form Kaggle wants it: "{:03d}_{}_{},{}"
	for new_line in patched_to_submission_lines(kaggle_pred, i+1):
		lines.append(new_line)
	
	break

# Create submission file
with open('data/submissions/submission_pytorch.csv', 'w') as f:
    f.write('id,prediction\n')
    f.writelines('{}\n'.format(line) for line in lines)

print("Predictions done.")


