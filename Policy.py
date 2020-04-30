import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import random
import copy
import Grid

class Policy():
	def __init__(self,grid,individual_types, parameter):
		self.number_of_actions=-1

	def do_action(self,grid,action_no):
		return grid

class Quarantine_area(Policy):
	def __init__(self, grid, individual_types, max_quarantine_distance, cost, valid_actions=None):
		if max_quarantine_distance<-1:
			print("Error: Not a valid bound for quarantine distance!")
			return None
		self.max_quarantine_distance=max_quarantine_distance
		self.number_of_actions = max_quarantine_distance+2 
		self.cost=cost
		self.policy_name="Quarantine"

		if valid_actions==None:
			self.valid_actions=[]
			for i in range(self.number_of_actions):
				self.valid_actions.append(i-1)

		else: self.valid_actions=valid_actions

	def do_action(self,grid,action_no):
		if action_no==-1:
			#Null policy where nothing happens
			return 0

		infected=[]
		for i in range(grid.grid_size):
			for j in range(grid.grid_size):
				if grid.agent_grid[i][j].individual_type=='Infected':
					infected.append((i,j))

		quarantined_set=self.neighbours_in_dist(grid,infected,action_no)
		for (x,y) in quarantined_set:
			grid.agent_grid[x][y].policy_state['quarantined']=True

		return self.cost*len(quarantined_set)

	def neighbours_in_dist(self,grid,infected,dist):
		quarantined_set=set(copy.deepcopy(infected))
		current_layer=infected

		for i in range(dist):
			temp_layer=set([])
			for (x,y) in current_layer:
				agent=grid.agent_grid[x][y]
				for n in agent.neighbours:
					coordinates=agent.coordinates
					quarantined_set.add(coordinates)
					temp_layer.add(coordinates)
			current_layer=temp_layer
		return quarantined_set



class Vaccinate_block(Policy):
	def __init__(self,grid,individual_types, block_size, cost,valid_actions=None):
		if grid.grid_size%block_size!=0:
			print("Error: Not a valid block size!")
			return None
		self.block_size=block_size
		self.number_of_actions = (int)(grid.grid_size/block_size)**2 
		self.cost=cost
		self.policy_name="Vaccinate"

		if valid_actions==None:
			self.valid_actions=[]
			for i in range(self.number_of_actions+1):
				self.valid_actions.append(i)

		else: self.valid_actions=valid_actions

	def do_action(self,grid,action_no):
		if action_no==-1 or action_no==self.number_of_actions:
			#Null policy where nothing happens
			return 0

		self.valid_actions.remove(action_no)
		no_of_blocks_in_row_or_col=(int)(grid.grid_size/self.block_size)
		row_start = self.block_size*(int)(action_no/no_of_blocks_in_row_or_col)
		col_start = self.block_size*(int)(action_no%no_of_blocks_in_row_or_col)

		for i in range(self.block_size):
			for j in range(self.block_size):
				grid.convert_type(i+row_start, j+col_start, 'Vaccinated')

		return self.cost

class Vaccinate_lines(Policy):
	def __init__(self,grid,individual_types, block_size, cost):
		if grid.grid_size%block_size!=0:
			print("Error: Not a valid block size!")
			return None
		self.block_size=block_size
		self.number_of_actions = (int)(grid.grid_size/block_size)**2 +1
		self.cost=cost
		self.valid_actions=[]
		for i in range(self.number_of_actions):
			self.valid_actions.append(i)

	def do_action(self,grid,action_no):
		if action_no==-1 or action_no==self.number_of_actions-1:
			#Null policy where nothing happens
			return 0
		self.valid_actions.remove(action_no)
		no_of_blocks_in_row_or_col=(int)(grid.grid_size/self.block_size)
		row_start = self.block_size*(int)(action_no/no_of_blocks_in_row_or_col)
		col_start = self.block_size*(int)(action_no%no_of_blocks_in_row_or_col)

		for i in range(self.block_size):
			for j in range(self.block_size):
				grid.convert_type(i+row_start, j+col_start, 'Vaccinated')

		return self.cost
