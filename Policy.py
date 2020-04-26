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

class Vaccinate_block(Policy):
	def __init__(self,grid,individual_types, block_size, cost):
		if grid.grid_size%block_size!=0:
			print("Error: Not a valid block size!")
			return None
		self.block_size=block_size
		self.number_of_actions = (int)(grid.grid_size/block_size)**2 
		self.cost=cost
		self.valid_actions=[]
		for i in range(self.number_of_actions+1):
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

class Vaccinate_lines(Policy):
	def __init__(self,grid,individual_types, block_size, cost):
		if grid.grid_size%block_size!=0:
			print("Error: Not a valid block size!")
			return None
		self.block_size=block_size
		self.number_of_actions = (int)(grid.grid_size/block_size)**2 
		self.cost=cost
		self.valid_actions=[]
		for i in range(self.number_of_actions+1):
			self.valid_actions.append(i)

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
