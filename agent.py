import numpy as np
import torch as T
import torch.nn as nn
import torch.autograd as autograd
from utils import ReplayBuffer
from network import ConvDQN, LinearDQN

class Agent():
	def __init__(self, env, network, learning_rate, gamma, buffer_size, eps_max, eps_min, eps_dec):
		self.env = env
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.replay_buffer = ReplayBuffer(max_size=buffer_size, input_shape = env.env_shape)
		self.eps=eps_max
		self.eps_min=eps_min
		self.eps_dec=eps_dec
		self.network = network

	def get_action(self, state):

		if(np.random.randn() <= self.eps):
			return self.env.sample_action()

		else:
			state = T.tensor(state, dtype=T.float).unsqueeze(0).to(self.model.device)
			actions = self.model.forward(state)
			return T.argmax(actions).item()

	def dec_eps(self):
		self.eps = max(self.eps_min, self.eps-self.eps_dec)

class Simple_Agent(Agent):
	"""
	This agent can handle the networks ConvDQN and LinearDQN. This agent uses a single DQN and a replay buffer for learning.
	"""
	def __init__(self, env, network, learning_rate, gamma, buffer_size, eps_max, eps_min, eps_dec):
		super(Simple_Agent, self).__init__(env, network, learning_rate, gamma, buffer_size, eps_max, eps_min, eps_dec)
		if self.network == "SimpleConvDQN":
			self.model = ConvDQN(env.env_shape, env.no_of_actions)
		elif self.network == "LinearDQN":
			self.model = LinearDQN(env.env_shape, env.no_of_actions)


	def update(self, batch_size):
		self.model.optimizer.zero_grad()

		batch = self.replay_buffer.sample(batch_size)
		states, actions, rewards, next_states, dones = batch
		states_t = T.tensor(states, dtype=T.float).to(self.model.device)
		actions_t = T.tensor(actions).to(self.model.device)
		rewards_t = T.tensor(rewards, dtype=T.float).to(self.model.device)
		next_states_t = T.tensor(next_states, dtype=T.float).to(self.model.device)

		curr_Q = self.model.forward(states_t).gather(1, actions_t.unsqueeze(1))
		curr_Q = curr_Q.squeeze(1)
		next_Q = self.model.forward(next_states_t)
		max_next_Q = T.max(next_Q, 1)[0]
		expected_Q = rewards_t + self.gamma * max_next_Q


		loss = self.model.MSE_loss(curr_Q, expected_Q).to(self.model.device)

		loss.backward()
		self.model.optimizer.step()

		self.dec_eps()
