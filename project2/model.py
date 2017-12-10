import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CNN(nn.Module):
	def __init__(self, learning_rate):
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

		self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)


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

	# predict is doing 'sigmoid' because we are using BCEWithLogitsLoss as a loss_function
	def predict(self, input):
		return F.sigmoid(self.forward(input))

class CompleteCNN(nn.Module):
	def __init__(self, learning_rate, activation):
		super(CompleteCNN, self).__init__()

		self.loss_function = nn.BCEWithLogitsLoss()

		# Could use PReLU instead (better)

		if activation == 'relu':
			self.act = nn.ReLU(inplace=True)
		elif activation == 'leaky_relu':
			self.act = nn.LeakyReLU(inplace=True)
		elif activation == 'prelu':
			self.act = nn.PReLU()

		self.in_conv = nn.Sequential(
			nn.Conv2d(3, 32, 3, padding=1),
			self.act)

		self.down1 = nn.Sequential(
			nn.BatchNorm2d(32),
			nn.Conv2d(32, 32, 3, padding=1),
			self.act,
			nn.BatchNorm2d(32),
			nn.Conv2d(32, 32, 3, padding=1),
			self.act)

		self.down2 = nn.Sequential(
			nn.MaxPool2d(2, stride=2),
			nn.BatchNorm2d(32),
			nn.Conv2d(32, 32, 3, padding=1),
			self.act,
			nn.BatchNorm2d(32),
			nn.Conv2d(32, 32, 3, padding=1),
			self.act,
			nn.BatchNorm2d(32),
			nn.Conv2d(32, 32, 3, padding=1),
			self.act)

		self.down3 = nn.Sequential(
			nn.MaxPool2d(2, stride=2),
			nn.BatchNorm2d(32),
			nn.Conv2d(32, 32, 3, padding=1),
			self.act,
			nn.BatchNorm2d(32),
			nn.Conv2d(32, 32, 3, padding=1),
			self.act,
			nn.BatchNorm2d(32),
			nn.Conv2d(32, 32, 3, padding=1),
			self.act)

		self.down4 = nn.Sequential(
			nn.MaxPool2d(2, stride=2),
			nn.BatchNorm2d(32),
			nn.Conv2d(32, 32, 3, padding=1),
			self.act,
			nn.BatchNorm2d(32),
			nn.Conv2d(32, 32, 3, padding=1),
			self.act,
			nn.BatchNorm2d(32),
			nn.Conv2d(32, 32, 3, padding=1),
			self.act)

		self.up1 = nn.Sequential(
			nn.MaxPool2d(2, stride=2),
			nn.BatchNorm2d(32),
			nn.Conv2d(32, 32, 3, padding=1),
			self.act,
			nn.BatchNorm2d(32),
			nn.Conv2d(32, 32, 3, padding=1),
			self.act,
			nn.BatchNorm2d(32),
			nn.ConvTranspose2d(32, 32, 2, stride=2),
			self.act)

		self.up2 = nn.Sequential(
			nn.BatchNorm2d(64),
			nn.Conv2d(64, 48, 3, padding=1),
			self.act,
			nn.BatchNorm2d(48),
			nn.Conv2d(48, 32, 3, padding=1),
			self.act,
			nn.BatchNorm2d(32),
			nn.ConvTranspose2d(32, 32, 2, stride=2),
			self.act)

		self.up3 = nn.Sequential(
			nn.BatchNorm2d(64),
			nn.Conv2d(64, 48, 3, padding=1),
			self.act,
			nn.BatchNorm2d(48),
			nn.Conv2d(48, 32, 3, padding=1),
			self.act,
			nn.BatchNorm2d(32),
			nn.ConvTranspose2d(32, 32, 2, stride=2),
			self.act)

		self.up4 = nn.Sequential(
			nn.BatchNorm2d(64),
			nn.Conv2d(64, 48, 3, padding=1),
			self.act,
			nn.BatchNorm2d(48),
			nn.Conv2d(48, 32, 3, padding=1),
			self.act,
			nn.BatchNorm2d(32),
			nn.ConvTranspose2d(32, 32, 2, stride=2),
			self.act)

		self.out_conv = nn.Sequential(
			nn.BatchNorm2d(64),
			nn.Conv2d(64, 48, 3, padding=1),
			self.act,
			nn.BatchNorm2d(48),
			nn.Conv2d(48, 32, 3, padding=1),
			self.act,
			nn.Conv2d(32, 1, 1))

		self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

	def forward(self, input):
		#print("1", input.size())
		in_conv = self.in_conv(input)
		#print("2", in_conv.size())
		down1 = self.down1(in_conv)
		#print("3", down1.size())
		down2 = self.down2(down1)
		#print("4", down2.size())
		down3 = self.down3(down2)
		#print("5", down3.size())
		down4 = self.down4(down3)
		#print("down4", down4.size())
		up1 = self.up1(down4)
		#print("6", up1.size())
		#print("new6", torch.cat([down4, up1], 1).size())
		up2 = self.up2(torch.cat([down4, up1], 1))
		#print("7", up2.size())
		#print("new7", torch.cat([down3, up2], 1).size())
		up3 = self.up3(torch.cat([down3, up2], 1))
		#print("8", up3.size())
		#print("new8", torch.cat([down2, up3], 1).size())
		up4 = self.up4(torch.cat([down2, up3], 1))
		#print("9", up4.size())
		#print("new9", torch.cat([down1, up4], 1).size())
		out_conv = self.out_conv(torch.cat([down1, up4], 1))
		#print("10", out_conv.size())

		return out_conv

	def step(self, input, target):
		self.train()
		self.zero_grad()
		out = self.forward(input)
		loss = self.loss_function(out, target)
		loss.backward()
		self.optimizer.step()

		return loss.data[0]

	# predict is doing 'sigmoid' because we are using BCEWithLogitsLoss as a loss_function
	def predict(self, input):
		return F.sigmoid(self.forward(input))

class SimpleCNN(nn.Module):
	def __init__(self, learning_rate):
		super(SimpleCNN, self).__init__()

		self.loss_function = nn.BCEWithLogitsLoss()

		self.layer1 = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=5, padding=2),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(2))

		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=5, padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2))

		self.fc1 = nn.Sequential(
			nn.Linear(20*20*32, 512),
			nn.ReLU())
		self.fc2 = nn.Linear(512, 2)

		self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
	def forward(self, x):
		out = self.layer1(x)
		print(1)
		print(out.size())
		out = self.layer2(out)
		print(2)
		print(out.size())
		out = out.view(out.size(0), -1)
		print(3)
		print(out.size())
		out = self.fc1(out)
		print(4)
		print(out.size())
		out = self.fc2(out)
		print(5)
		print(out.size())
		return out

	def step(self, input, target):
		self.train()
		self.zero_grad()
		out = self.forward(input)
		loss = self.loss_function(out, target)
		loss.backward()
		self.optimizer.step()

		return loss.data[0]

	# predict is doing 'sigmoid' because we are using BCEWithLogitsLoss as a loss_function
	def predict(self, input):
		return F.sigmoid(self.forward(input))
