import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LinearDQN(nn.Module):

	def __init__(self, input_dim, output_dim):
		super().__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim

		self.fc = nn.Sequential(
			nn.Linear(self.input_dim[1]*self.input_dim[2], 256),
			nn.ReLU(),
			nn.Linear(256, 512),
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, self.output_dim)
		)

		self.optimizer = optim.Adam(self.parameters())
		self.MSE_loss = nn.MSELoss()
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state):
		qvals = self.fc(T.flatten(state,start_dim=1))
		return qvals


class ConvDQN(nn.Module):

	def __init__(self, input_dim, output_dim):
		super().__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim

		self.conv = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, stride=1),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(64, 16, kernel_size=3, stride=1),
			nn.ReLU()
		)

		self.fc_input_dim = self.feature_size()
		self.fc = nn.Sequential(
			nn.Linear(self.fc_input_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, self.output_dim)
		)

		self.optimizer = optim.Adam(self.parameters())
		self.MSE_loss = nn.MSELoss()
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def feature_size(self):
		state = T.zeros(1, *self.input_dim)
		dims = self.conv(state)
		return int(np.prod(dims.size()))

	def forward(self, state):
		features = self.conv(state)
		features = features.view(features.size(0), -1)
		qvals = self.fc(features)
		return qvals


class DuelingDQN(nn.Module):

	def __init__(self, input_dim, output_dim):
		super().__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim

		self.conv = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, stride=1),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(64, 16, kernel_size=3, stride=1),
			nn.ReLU()
		)

		self.fc_input_dim = self.feature_size()
		self.fc1 = nn.Sequential(
			nn.Linear(self.fc_input_dim, 64),
			nn.ReLU()
		)
		self.V = nn.Linear(64, 1)
		self.A = nn.Linear(64, self.output_dim)

		self.optimizer = optim.Adam(self.parameters())
		self.MSE_loss = nn.MSELoss()
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def feature_size(self):
		state = T.zeros(1, *self.input_dim)
		dims = self.conv(state)
		return int(np.prod(dims.size()))

	def forward(self, state):
		features = self.conv(state)
		features = features.view(features.size(0), -1)
		outs = self.fc1(features)
		V = self.V(outs)
		A = self.A(outs)

		return V, A
