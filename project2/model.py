import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from parameters import LEARNING_RATE

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()

		self.loss_function = nn.BCEWithLogitsLoss()

		# Could use PReLU instead (better)
		self.act = nn.ReLU(inplace=True)

		self.in_conv = nn.Sequential(
						nn.Conv2d(3, 32, 3, padding=1), # from 3 to 32 channels (3x3)
						nn.BatchNorm2d(32), # seems to be good, read about it
						self.act, # ReLU
						nn.Conv2d(32, 32, 3, padding=1),
						nn.BatchNorm2d(32),
						self.act
					)

		self.down1 = nn.Sequential(
						nn.MaxPool2d(2),
						nn.Conv2d(32, 64, 3, padding=1),
						nn.BatchNorm2d(64),
						self.act,
						nn.Conv2d(64, 64, 3, padding=1),
						nn.BatchNorm2d(64),
						self.act
					)

		self.up1 = nn.Sequential(
						nn.ConvTranspose2d(64, 64, 2, stride=2),
						nn.Conv2d(64, 32, 3, padding=1),
						nn.BatchNorm2d(32),
						self.act,
						nn.Conv2d(32, 32, 3, padding=1),
						nn.BatchNorm2d(32),
						self.act
					)

		self.out_conv = nn.Sequential(
						nn.Conv2d(32, 1, 3, padding=1),
						nn.BatchNorm2d(1),
						self.act,
						nn.Conv2d(1, 1, 1)
					)

		self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)


	def forward(self, input):
		in_conv = self.in_conv(input)
		down1 = self.down1(in_conv)
		up1 = self.up1(down1)
		out_conv = self.out_conv(up1)

		return out_conv


	def step(self, input, target):
		self.train()
		self.zero_grad()
		out = self.forward(input)
		loss = self.loss_function(out, target)
		loss.backward()
		self.optimizer.step()

		return loss.data[0]

class DummyCNN(nn.Module):
	def __init__(self):
		super(DummyCNN, self).__init__()

		self.loss_function = nn.BCEWithLogitsLoss()

		# Could use PReLU instead (better)
		self.act = nn.ReLU(inplace=True)

		self.in_conv = nn.Sequential(
						nn.Conv2d(3, 32, 3, padding=1), # from 3 to 32 channels (3x3)
						nn.BatchNorm2d(32), # seems to be good, read about it
						self.act, # ReLU,
						nn.MaxPool2d(2)
					)

		self.down1 = nn.Sequential(
						nn.Conv2d(32, 64, 3, padding=1), # from 3 to 32 channels (3x3)
						nn.BatchNorm2d(64), # seems to be good, read about it
						self.act, # ReLU,
						nn.MaxPool2d(2),
					)

		self.up1 = nn.Sequential(
						nn.Linear(64, 512),
						self.act
					)

		self.out_conv = nn.Sequential(
						nn.Linear(512, 2)
					)

		self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)


	def forward(self, input):
		x = self.in_conv(input)
		x = self.down1(x)
		x = self.up1(x)
		x = self.out_conv(x)

		return x


	def step(self, input, target):
		self.train()
		self.zero_grad()
		out = self.forward(input)
		loss = self.loss_function(out, target)
		loss.backward()
		self.optimizer.step()

		return loss.data[0]
