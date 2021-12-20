import numpy as np
import torch as T
import torch.nn as nn
from utils import ReplayBuffer
from network import ConvDQN, LinearDQN, DuelingDQN
import os

class Agent():
	def __init__(self, env, network, learning_rate, gamma, eps_max, eps_min, eps_dec, chkpt_dir, name):
		self.env = env
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.eps=eps_max
		self.eps_min=eps_min
		self.eps_dec=eps_dec
		self.network = network
		self.checkpoint_dir = chkpt_dir
		if not os.path.isdir(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)
		self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

	def dec_eps(self):
		self.eps = max(self.eps_min, self.eps-self.eps_dec)

	def learn(self, *args, **kwargs):
		raise NotImplementedError

class Simple_DQNAgent(Agent):
	"""
	This agent can handle the networks ConvDQN and LinearDQN. This agent uses a single DQN and a replay buffer for learning.
	"""
	def __init__(self, env, network, learning_rate, gamma, eps_max, eps_min, eps_dec, buffer_size, chkpt_dir, name):
		super().__init__(env, network, learning_rate, gamma, eps_max, eps_min, eps_dec, chkpt_dir, name)

		if self.network == "SimpleConvDQN":
			self.model = ConvDQN(env.env_shape, env.no_of_actions)
		elif self.network == "LinearDQN":
			self.model = LinearDQN(env.env_shape, env.no_of_actions)

		self.replay_buffer = ReplayBuffer(max_size=buffer_size, input_shape = env.env_shape)

	def get_action(self, state):

		if(np.random.randn() <= self.eps):
			return self.env.sample_action()

		else:
			state = T.tensor(state, dtype=T.float).unsqueeze(0).to(self.model.device)
			actions = self.model.forward(state)
			return T.argmax(actions).item()

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

	def learn(self,state, action, reward, next_state, done, batch_size):
		self.replay_buffer.store_transition(state, action, reward, next_state, done)

		if len(self.replay_buffer) > batch_size:
			self.update(batch_size)

	def save_models(self):
		print('... saving checkpoint ...')
		T.save(self.model.state_dict(), self.checkpoint_file)

	def load_models(self):
		print('... loading checkpoint ...')
		self.model.load_state_dict(T.load(self.checkpoint_file))


class DQNAgent(Agent):
	"""
	Uses a replay buffer and has two DQNs, one that is used to get best actions and updated every step and the other, a target network,
	used to compute the target Q value every step. This target network is only updated with the first DQN only after a fixed number of steps.
	"""
	def __init__(self, env, network, learning_rate, gamma, eps_max, eps_min, eps_dec, buffer_size, replace_cnt, chkpt_dir, name):
		super().__init__(env, network, learning_rate, gamma, eps_max, eps_min, eps_dec, chkpt_dir, name)

		self.replay_buffer = ReplayBuffer(max_size=buffer_size, input_shape = env.env_shape)

		self.learn_step_counter = 0
		self.replace_cnt = replace_cnt
		self.q_eval = ConvDQN(env.env_shape, env.no_of_actions)
		self.q_target = ConvDQN(env.env_shape, env.no_of_actions)


	def get_action(self, state):

		if(np.random.randn() <= self.eps):
			return self.env.sample_action()

		else:
			state = T.tensor(state, dtype=T.float).unsqueeze(0).to(self.q_eval.device)
			actions = self.q_eval.forward(state)
			return T.argmax(actions).item()

	def replace_target_network(self):
		if self.learn_step_counter % self.replace_cnt == 0:
			self.q_target.load_state_dict(self.q_eval.state_dict())

	def get_batch_tensors(self, batch_size):
		batch = self.replay_buffer.sample(batch_size)
		states, actions, rewards, next_states, dones = batch
		states_t = T.tensor(states, dtype=T.float).to(self.q_eval.device)
		actions_t = T.tensor(actions).to(self.q_eval.device)
		rewards_t = T.tensor(rewards, dtype=T.float).to(self.q_eval.device)
		next_states_t = T.tensor(next_states, dtype=T.float).to(self.q_eval.device)
		return states_t, actions_t, rewards_t, next_states_t

	def update(self, batch_size):

		states_t, actions_t, rewards_t, next_states_t = self.get_batch_tensors(batch_size)
		self.q_eval.optimizer.zero_grad()

		self.replace_target_network()

		indices = np.arange(batch_size)
		curr_Q = self.q_eval.forward(states_t)[indices, actions_t]
		max_next_Q = self.q_target.forward(next_states_t).max(1)[0]
		expected_Q = rewards_t + self.gamma * max_next_Q

		loss = self.q_eval.MSE_loss(curr_Q, expected_Q).to(self.q_eval.device)

		loss.backward()
		self.q_eval.optimizer.step()
		self.learn_step_counter += 1

		self.dec_eps()

	def learn(self,state, action, reward, next_state, done, batch_size):
		self.replay_buffer.store_transition(state, action, reward, next_state, done)

		if len(self.replay_buffer) > batch_size:
			self.update(batch_size)

	def save_models(self):
		print('... saving checkpoint ...')
		T.save(self.q_eval.state_dict(), self.checkpoint_file+'_qeval')
		T.save(self.q_target.state_dict(), self.checkpoint_file+'_qtarget')

	def load_models(self):
		print('... loading checkpoint ...')
		self.q_eval.load_state_dict(T.load(self.checkpoint_file+'_qeval'))
		self.q_target.load_state_dict(T.load(self.checkpoint_file+'_qtarget'))


class DoubleDQNAgent(DQNAgent):
	def __init__(self, env, network, learning_rate, gamma, eps_max, eps_min, eps_dec, buffer_size, replace_cnt, chkpt_dir, name):
		super().__init__(env, network, learning_rate, gamma, eps_max, eps_min, eps_dec, buffer_size, replace_cnt, chkpt_dir, name)


	def update(self, batch_size):

		states_t, actions_t, rewards_t, next_states_t = self.get_batch_tensors(batch_size)

		self.q_eval.optimizer.zero_grad()

		self.replace_target_network()

		indices = np.arange(batch_size)
		curr_Q = self.q_eval.forward(states_t)[indices, actions_t]
		max_indices = self.q_eval.forward(next_states_t).max(1)[1]
		max_next_Q = self.q_target.forward(next_states_t)[indices, max_indices]
		expected_Q = rewards_t + self.gamma * max_next_Q

		loss = self.q_eval.MSE_loss(curr_Q, expected_Q).to(self.q_eval.device)

		loss.backward()
		self.q_eval.optimizer.step()
		self.learn_step_counter += 1

		self.dec_eps()


class DuelingDQNAgent(DQNAgent):
	def __init__(self, env, network, learning_rate, gamma, eps_max, eps_min, eps_dec, buffer_size, replace_cnt, chkpt_dir, name):
		super().__init__(env, network, learning_rate, gamma, eps_max, eps_min, eps_dec, buffer_size, replace_cnt, chkpt_dir, name)

		self.q_eval = DuelingDQN(env.env_shape, env.no_of_actions)
		self.q_target = DuelingDQN(env.env_shape, env.no_of_actions)


	def get_action(self, state):

		if(np.random.randn() <= self.eps):
			return self.env.sample_action()

		else:
			state = T.tensor(state, dtype=T.float).unsqueeze(0).to(self.q_eval.device)
			_, advantage = self.q_eval.forward(state)
			return T.argmax(advantage).item()

	def update(self, batch_size):

		states_t, actions_t, rewards_t, next_states_t = self.get_batch_tensors(batch_size)
		self.q_eval.optimizer.zero_grad()

		self.replace_target_network()

		indices = np.arange(batch_size)
		Vs, As = self.q_eval.forward(states_t)
		curr_Q = T.add(Vs, (As - As.mean(dim=1, keepdim=True)))[indices, actions_t]

		Vns, Ans = self.q_target.forward(next_states_t)
		max_next_Q = T.add(Vs, (As - As.mean(dim=1, keepdim=True))).max(1)[0]

		expected_Q = rewards_t + self.gamma * max_next_Q

		loss = self.q_eval.MSE_loss(curr_Q, expected_Q).to(self.q_eval.device)

		loss.backward()
		self.q_eval.optimizer.step()
		self.learn_step_counter += 1

		self.dec_eps()


class DuelingDoubleDQNAgent(DuelingDQNAgent):
	def __init__(self, env, network, learning_rate, gamma, eps_max, eps_min, eps_dec, buffer_size, replace_cnt, chkpt_dir, name):
		super().__init__(env, network, learning_rate, gamma, eps_max, eps_min, eps_dec, buffer_size, replace_cnt, chkpt_dir, name)


	def update(self, batch_size):

		states_t, actions_t, rewards_t, next_states_t = self.get_batch_tensors(batch_size)
		self.q_eval.optimizer.zero_grad()

		self.replace_target_network()

		indices = np.arange(batch_size)
		Vs, As = self.q_eval.forward(states_t)
		curr_Q = T.add(Vs, (As - As.mean(dim=1, keepdim=True)))[indices, actions_t]

		Vs_, As_ = self.q_eval.forward(next_states_t)
		max_indices = T.add(Vs_, (As_ - As_.mean(dim=1, keepdim=True))).max(1)[1]
		Vns, Ans = self.q_target.forward(next_states_t)
		max_next_Q = T.add(Vns, (Ans - Ans.mean(dim=1, keepdim=True)))[indices, max_indices]

		expected_Q = rewards_t + self.gamma * max_next_Q

		loss = self.q_eval.MSE_loss(curr_Q, expected_Q).to(self.q_eval.device)

		loss.backward()
		self.q_eval.optimizer.step()
		self.learn_step_counter += 1

		self.dec_eps()

def get_agent(env, args, name):
	if args.network in ["LinearDQN", "SimpleConvDQN"]:
		return Simple_DQNAgent(env, args.network, args.learning_rate, args.gamma, args.eps_max, args.eps_min, args.eps_dec, args.max_buffer_size, args.path, name)

	elif args.network == "ConvDQN":
		return DQNAgent(env, args.network, args.learning_rate, args.gamma, args.eps_max, args.eps_min, args.eps_dec, args.max_buffer_size, args.update_steps, args.path, name)

	elif args.network == "DoubleDQN":
		return DoubleDQNAgent(env, args.network, args.learning_rate, args.gamma, args.eps_max, args.eps_min, args.eps_dec, args.max_buffer_size, args.update_steps, args.path, name)

	elif args.network == "DuelingDQN":
		return DuelingDQNAgent(env, args.network, args.learning_rate, args.gamma, args.eps_max, args.eps_min, args.eps_dec, args.max_buffer_size, args.update_steps, args.path, name)

	elif args.network == "DuelingDoubleDQN":
		return DuelingDoubleDQNAgent(env, args.network, args.learning_rate, args.gamma, args.eps_max, args.eps_min, args.eps_dec, args.max_buffer_size, args.update_steps, args.path, name)

	else:
		raise Exception("Enter Valid Network! Choose from : LinearDQN, SimpleConvDQN, ConvDQN, DoubleDQN, DuelingDQN, DuelingDoubleDQN")
