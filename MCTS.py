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
			
			#gridtable=copy.deepcopy(state_obj.grid.grid)
			#grid=Grid.Grid(gridtable,state_obj.individual_types)
			#policy=Policy.Vaccinate_block(grid, state_obj.individual_types,state_obj.policy.block_size,state_obj.policy.cost,copy.deepcopy(state_obj.policy.valid_actions))
			#temp_obj= Simulate.Simulate(state_obj.transmission_prob,state_obj.individual_types,grid,policy)
			temp_obj=state_obj.copy_cstr()

			temp_obj.simulate_day(action)
			reward=temp_obj.simulate_till_end(self.reward_fn)
			total_score+=reward
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
			print("Action taken: ",action)
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

def scenario():
	def p_infection(day,global_state,my_agent,neighbour_agents):  # probability of infectiong neighbour
		p_inf=0.3
		p_not_inf=1
		for nbr_agent in neighbour_agents:
			if nbr_agent.individual_type in ['Infected','Asymptomatic'] and not nbr_agent.policy_state['quarantined']:
				p_not_inf*=(1-p_inf)

		return 1 - p_not_inf

		return 0.3

	def p_standard(p):
		def p_fn(day,global_state,a1,nbrs):
			return p
		return p_fn


	individual_types=['Susceptible','Infected','Recovered','Vaccinated','Asymptomatic','Exposed']
	color_list=['green', 'black','red','blue','grey','white']

	transmission_prob={}
	for t in individual_types:
		transmission_prob[t]={}

	for t1 in individual_types:
		for t2 in individual_types:
			transmission_prob[t1][t2]=p_standard(0)

	transmission_prob['Susceptible']['Exposed']= p_infection
	transmission_prob['Exposed']['Infected']= p_standard(1)
	transmission_prob['Exposed']['Asymptomatic']= p_standard(0)
	transmission_prob['Infected']['Recovered']= p_standard(0.2)
	transmission_prob['Asymptomatic']['Recovered']= p_standard(0.2)
	transmission_prob['Recovered']['Susceptible']= p_standard(0)
	gridtable =np.zeros((12,12))
	gridtable[1][1]=1
	gridtable[5][1]=1
	gridtable[3][7]=1
	gridtable[2][1]=1
	gridtable[3][6]=4
	gridtable[7][3]=4
	gridtable[8][7]=4
	gridtable[11][3]=1

	grid=Grid.Grid(gridtable,individual_types)
	policy=Policy.Vaccinate_block(grid, individual_types,3,0)
	#policy=Policy.Quarantine_area(grid, individual_types, 2, 0)
	sim_obj= Simulate.Simulate(transmission_prob,individual_types,grid,policy)
	
	def reward_fn(days,no_infected):
		return -days

	mc_obj=MCTS(sim_obj,300,reward_fn)
	mc_obj.run_game(color_list)

#scenario()  


def scenario2():
	def p_infection(day,global_state,my_agent,neighbour_agents):  # probability of infectiong neighbour
		p_inf=0.3
		p_not_inf=1
		for nbr_agent in neighbour_agents:
			if nbr_agent.individual_type in ['Infected','Asymptomatic'] and not nbr_agent.policy_state['quarantined']:
				p_not_inf*=(1-p_inf)

		return 1 - p_not_inf


	def p_standard(p):
		def p_fn(day,global_state,a1,nbrs):
			return p
		return p_fn


	individual_types=['Susceptible','Infected','Recovered','Vaccinated','Mild','Exposed','Asymptomatic']
	color_list=['white', 'black','red','pink','grey','black','black']

	transmission_prob={}
	for t in individual_types:
		transmission_prob[t]={}

	for t1 in individual_types:
		for t2 in individual_types:
			transmission_prob[t1][t2]=p_standard(0)

	transmission_prob['Susceptible']['Infected']= p_infection
	transmission_prob['Susceptible']['Mild']= p_infection
	#transmission_prob['Exposed']['Infected']= p_standard(1)
	#transmission_prob['Exposed']['Asymptomatic']= p_standard(0)
	transmission_prob['Infected']['Recovered']= p_standard(0.2)
	transmission_prob['Mild']['Recovered']= p_standard(0.6)
	transmission_prob['Recovered']['Susceptible']= p_standard(0)
	gridtable =np.zeros((12,12))
	gridtable[1][1]=1
	gridtable[5][1]=1
	gridtable[3][7]=1
	gridtable[2][1]=1
	gridtable[5][6]=1
	gridtable[4][4]=1
	gridtable[2][1]=1
	gridtable[3][6]=1
	gridtable[7][3]=1
	gridtable[8][7]=4
	gridtable[11][3]=1

	grid=Grid.Grid(gridtable,individual_types)
	policy=Policy.Vaccinate_block(grid, individual_types,3,0)
	#policy=Policy.Quarantine_area(grid, individual_types, 2, 0)
	sim_obj= Simulate.Simulate(transmission_prob,individual_types,grid,policy)
	
	def reward_fn(days,no_infected):
		return -days

	mc_obj=MCTS(sim_obj,100,reward_fn)
	mc_obj.run_game(color_list)

scenario2()











