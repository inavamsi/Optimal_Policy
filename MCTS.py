import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import random
import copy
import Grid
import Policy
import Simulate
import time

class MCTS():
	def __init__(self,start_state_obj, number_of_runs, reward_fn):
		self.current_state_obj = start_state_obj
		self.number_of_runs=number_of_runs
		self.reward_fn=reward_fn

	def get_score_of_action(self,state_obj,action):
		total_score=0
		for i in range(self.number_of_runs):
			temp_obj=copy.deepcopy(state_obj)
			temp_obj.simulate_day(action)
			total_score+=temp_obj.simulate_till_end(self.reward_fn)
		return total_score

	def get_action(self,state_obj):
		max_score=-np.inf
		max_action=-1
		for action in state_obj.policy.valid_actions:
			action_score=self.get_score_of_action(state_obj,action)
			if action_score>max_score:
				max_score=action_score
				max_action=action
			
		return max_action

	def run_game(self,color_list):
		ts=time.time()
		while(self.current_state_obj.grid.current_types_pop['Infected']!=0):
			action=self.get_action(self.current_state_obj)
			self.current_state_obj.simulate_day(action)

		print("Time taken: ",time.time()-ts)
		print("Days taken :",self.current_state_obj.day)
		print("Total Infected Days :",self.current_state_obj.total_infected_days)
		self.current_state_obj.grid.animate(False,color_list,1)
		self.current_state_obj.grid.plot_time_series()


		while(1):
			run_again=1
			input(run_again)
			if run_again==1:
				self.current_state_obj.grid.animate(False,color_list,1)
			else:
				break


def scenario1():
	def p_infection(day,cur_type_pop,state):
		return 0.5
	def p_recovery(day,cur_type_pop,state):
		return 0.2
	def p_unimmunisation(day,cur_type_pop,state):
		return 0
	individual_types=['Susceptible','Infected','Immune']
	color_list=['white', 'black','red']
	gridtable =np.zeros((12,12))
	gridtable[1][1]=1
	gridtable[5][1]=1
	gridtable[3][7]=1
	gridtable[2][1]=1
	gridtable[3][6]=1
	gridtable[7][3]=1
	gridtable[3][8]=2
	gridtable[2][8]=2
	gridtable[1][8]=2
	gridtable[0][8]=2
	gridtable[8][0]=2
	gridtable[8][1]=2
	gridtable[8][2]=2
	gridtable[8][3]=2
	grid=Grid.Grid(gridtable,individual_types)
	policy=Policy.Vaccinate_block(grid, individual_types,3,0)
	sim_obj= Simulate.Simulate(p_infection, p_recovery, p_unimmunisation,individual_types,grid,policy)
	
	def reward_fn(days,no_infected):
		return -days -0.01*no_infected

	mc_obj=MCTS(sim_obj,30,reward_fn)
	mc_obj.run_game(color_list)

def scenario2():
	def p_infection(day,cur_type_pop,state):
		return 0.5
	def p_recovery(day,cur_type_pop,state):
		return 0.2
	def p_unimmunisation(day,cur_type_pop,state):
		return 0
	individual_types=['Susceptible','Infected','Immune']
	color_list=['white', 'black','red']
	gridtable =np.zeros((12,12))
	gridtable[1][1]=1
	gridtable[5][1]=1
	gridtable[3][7]=1
	gridtable[2][1]=1
	gridtable[3][6]=1
	gridtable[7][3]=1
	gridtable[8][7]=1
	gridtable[11][3]=1

	grid=Grid.Grid(gridtable,individual_types)
	policy=Policy.Vaccinate_block(grid, individual_types,3,0)
	sim_obj= Simulate.Simulate(p_infection, p_recovery, p_unimmunisation,individual_types,grid,policy)
	
	def reward_fn(days,no_infected):
		return -no_infected

	mc_obj=MCTS(sim_obj,100,reward_fn)
	mc_obj.run_game(color_list)

def scenario3():
	def p_infection(day,cur_type_pop,state):
		return 0.5
	def p_recovery(day,cur_type_pop,state):
		return 0.2
	def p_unimmunisation(day,cur_type_pop,state):
		return 0
	individual_types=['Susceptible','Infected','Immune','Vaccinated']
	color_list=['white', 'black','red','blue']
	gridtable =np.zeros((12,12))
	gridtable[1][1]=1
	gridtable[5][1]=1
	gridtable[3][7]=1
	gridtable[2][1]=1
	gridtable[3][6]=1
	gridtable[7][3]=1
	gridtable[8][7]=1
	gridtable[11][3]=1

	grid=Grid.Grid(gridtable,individual_types)
	policy=Policy.Vaccinate_block(grid, individual_types,3,0)
	sim_obj= Simulate.Simulate(p_infection, p_recovery, p_unimmunisation,individual_types,grid,policy)
	
	def reward_fn(days,no_infected):
		return -days

	mc_obj=MCTS(sim_obj,200,reward_fn)
	mc_obj.run_game(color_list)
scenario3()  
















