#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
"""
    Convolutional Neural Network implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CNN(nn.Module):
	def __init__(self, learning_rate, activation, optimizer, momentum=0.9):
		super(CompleteCNN, self).__init__()

		self.loss_function = nn.BCEWithLogitsLoss()

		if activation == 'relu':
			self.act = nn.ReLU(inplace=True)
		elif activation == 'leaky_relu':
			self.act = nn.LeakyReLU(inplace=True)
		elif activation == 'prelu':
			self.act = nn.PReLU()

		if optimizer == 'Adam':
			self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
		elif optimizer == 'SGD':
			self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
		elif optimizer == 'SGD + Momentum':
			self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)

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

		self.out_conv = nn.Sequential(
			nn.BatchNorm2d(64),
			nn.Conv2d(64, 48, 3, padding=1),
			self.act,
			nn.BatchNorm2d(48),
			nn.Conv2d(48, 32, 3, padding=1),
			self.act,
			nn.Conv2d(32, 1, 1))

	def forward(self, input):
		in_conv = self.in_conv(input)
		down1 = self.down1(in_conv)
		down2 = self.down2(down1)
		down3 = self.down3(down2)
		up1 = self.up1(down3)
		up2 = self.up2(torch.cat([down3, up1], 1))
		up3 = self.up3(torch.cat([down2, up2], 1))
		out_conv = self.out_conv(torch.cat([down1, up3], 1))

		return out_conv

	def step(self, input, target):
		self.train()
		self.zero_grad()
		out = self.forward(input)
		loss = self.loss_function(out, target)
		loss.backward()
		self.optimizer.step()

		return loss.data[0]

	def predict(self, input):
		return F.sigmoid(self.forward(input))