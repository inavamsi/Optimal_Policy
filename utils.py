import matplotlib.pyplot as plt
from matplotlib import colors
from collections import deque
import numpy as np
import random
import time

# Replay Buffer
class ReplayBuffer():
	def __init__(self, max_size, input_shape):
		self.mem_size = max_size
		self.mem_cntr = 0
		self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
		self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
		self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
		self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
		self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

	def store_transition(self, state, action, reward, next_state, done):
		index = self.mem_cntr % self.mem_size
		self.state_memory[index] = state
		self.new_state_memory[index] = next_state
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.terminal_memory[index] = done

		self.mem_cntr += 1

	def sample(self, batch_size):
		max_mem = min(self.mem_cntr, self.mem_size)
		batch = np.random.choice(max_mem, batch_size, replace = False)

		states = self.state_memory[batch]
		new_states = self.new_state_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		dones = self.terminal_memory[batch]

		return states, actions, rewards, new_states, dones

	def __len__(self):
		return min(self.mem_cntr, self.mem_size)

# Plotting
def plot_time_series(grid):
	for t in grid.individual_types:
		plt.plot(grid.type_timeseries[t])

	plt.title('Population timeline of different individual types')
	plt.legend(grid.individual_types,loc='upper right', shadow=True)
	plt.show()

def plot_grid(grid,gridlines,color_list):
	data=grid.grid
	n=grid.grid_size
	# create discrete colormap
	cmap = colors.ListedColormap(color_list)
	bounds=[-0.5]
	for i in range(grid.no_types):
		bounds.append(bounds[i]+1)
	norm = colors.BoundaryNorm(bounds, cmap.N)

	fig, ax = plt.subplots()
	ax.imshow(data, cmap=cmap, norm=norm)

	# draw gridlines
	if(gridlines):
		ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
		ax.set_xticks(np.arange(-.5, n, 1));
		ax.set_yticks(np.arange(-.5, n, 1));

	plt.pause(0.3)
	plt.close()

def animate(grid, gridlines,color_list, time):
	for g in grid.store:
		data=g
		n=grid.grid_size
		# create discrete colormap
		cmap = colors.ListedColormap(color_list)
		bounds=[-0.5]
		for i in range(grid.no_types):
			bounds.append(bounds[i]+1)
		norm = colors.BoundaryNorm(bounds, cmap.N)

		fig, ax = plt.subplots()
		ax.imshow(data, cmap=cmap, norm=norm)

		# draw gridlines
		if(gridlines):
			ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
			ax.set_xticks(np.arange(-.5, n, 1))
			ax.set_yticks(np.arange(-.5, n, 1))
		plt.pause(time)
		plt.close()

def plot_learning_curve(scores, eps_history):
	x = [i+1 for i in range(len(scores))]

	fig = plt.figure()
	ax = fig.add_subplot(111, label="1")
	ax2 = fig.add_subplot(111, label="2", frame_on=False)

	ax.plot(x, eps_history, color="C0")
	ax.set_xlabel("Training Steps", color="C0")
	ax.set_ylabel("Epsilon", color="C0")
	ax.tick_params(axis='x', colors="C0")
	ax.tick_params(axis='y', colors="C0")

	N = len(scores)
	running_avg = np.empty(N)
	for t in range(N):
		running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

	ax2.scatter(x, running_avg, color="C1")
	ax2.axes.get_xaxis().set_visible(False)
	ax2.yaxis.tick_right()
	ax2.set_ylabel('Score', color="C1")
	ax2.yaxis.set_label_position('right')
	ax2.tick_params(axis='y', colors="C1")
	plt.show()
