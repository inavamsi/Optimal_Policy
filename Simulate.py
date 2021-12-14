import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import random
import copy
import Grid
import Policy
import time

class Simulate():
	def __init__(self, transmission_prob, individual_types,grid,policy):
		self.individual_types=individual_types
		self.grid=grid
		self.transmission_prob=transmission_prob
		self.day=0
		self.policy=policy
		self.total_infected_days=0
		self.global_state='Normal'

	def copy_cstr(self):
		gridtable=copy.deepcopy(self.grid.grid)
		grid=Grid.Grid(gridtable,self.individual_types)
		if self.policy.policy_name=="Vaccinate":
			policy=Policy.Vaccinate_block(grid, self.individual_types,self.policy.block_size,self.policy.cost,copy.deepcopy(self.policy.valid_actions))
		if self.policy.policy_name=="Quarantine":
			policy=Policy.Vaccinate_block(grid, self.individual_types,self.policy.max_quarantine_distance,self.policy.cost,copy.deepcopy(self.policy.valid_actions))
		temp_obj= Simulate(self.transmission_prob,self.individual_types,grid,policy)

		for i in range(grid.grid_size):
			for j in range(grid.grid_size):
				temp_obj.grid.agent_grid[i][j].policy_state=copy.deepcopy(self.grid.agent_grid[i][j].policy_state)

		return temp_obj

	def simulate_day(self,action_no):
		self.policy.do_action(self.grid,action_no)
		grid=self.grid
		new_grid=copy.deepcopy(grid.grid)

		for i in range(grid.grid_size):
			for j in range(grid.grid_size):
				cur_agent=grid.agent_grid[i][j]

				conversion_type=self.find_conversion_type(cur_agent)
				new_grid[i][j]=grid.type_to_number[conversion_type]

		for i in range(grid.grid_size):
			for j in range(grid.grid_size):
				cur_type=grid.number_to_type[new_grid[i][j]]
				grid.convert_type(i,j,cur_type)

		self.day+=1
		grid.update_timeseries()
		if 'Infected' in self.individual_types:
			self.total_infected_days+=self.grid.current_types_pop['Infected']
		return

	def find_conversion_type(self,agent):
		my_type=agent.individual_type
		r=random.random()
		p=0
		for t in self.individual_types:
			p+=self.transmission_prob[my_type][t](self.day,self.global_state,agent,agent.neighbours)
			if r<p:
				return t

		return my_type

	def simulate_days(self,n):
		for i in range(n):
			self.simulate_day(-1)

	def simulate_till_end(self, reward_fn):
		no_infected=self.grid.current_types_pop['Infected']+self.grid.current_types_pop['Asymptomatic']+self.grid.current_types_pop['Exposed']
		total_infected_days=0
		while(no_infected>0):
			action_no=random.choice(self.policy.valid_actions)

			self.simulate_day(action_no)
			no_infected=self.grid.current_types_pop['Infected']+self.grid.current_types_pop['Asymptomatic']+self.grid.current_types_pop['Exposed']
			total_infected_days+=no_infected

		return reward_fn(self.day,total_infected_days)

if __name__ == "__main__":
	#Standard spread
	def p_infection(day,global_state,my_agent,neighbour_agents):  # probability of infectiong neighbour
		p_inf=0.5
		p_not_inf=1
		for nbr_agent in neighbour_agents:
			if nbr_agent.individual_type in ['Infected','Asymptomatic'] and not nbr_agent.policy_state['quarantined']:
				p_not_inf*=(1-p_inf)

		return 1 - p_not_inf


	def p_standard(p):
		def p_fn(day,global_state,a1,nbrs):
			return p
		return p_fn


	individual_types=['Susceptible','Infected','Recovered','Vaccinated','Asymptomatic','Exposed']
	color_list=['white', 'black','red','blue','blue','grey']

	transmission_prob={}
	for t in individual_types:
		transmission_prob[t]={}

	for t1 in individual_types:
		for t2 in individual_types:
			transmission_prob[t1][t2]=p_standard(0)

	transmission_prob['Susceptible']['Exposed']= p_infection
	transmission_prob['Exposed']['Infected']= p_standard(0.5)
	transmission_prob['Exposed']['Asymptomatic']= p_standard(0.3)
	transmission_prob['Infected']['Recovered']= p_standard(0.2)
	transmission_prob['Asymptomatic']['Recovered']= p_standard(0.2)
	transmission_prob['Recovered']['Susceptible']= p_standard(0)

	gridtable =np.zeros((50,50))
	gridtable[35][6]=1
	gridtable[22][43]=1
	gridtable[7][2]=1
	gridtable[15][2]=1
	grid=Grid.Grid(gridtable,individual_types)
	policy=Policy.Vaccinate_block(grid, individual_types,1,0)
	sim_obj= Simulate(transmission_prob,individual_types,grid,policy)

	def reward_fn(days,no_infected):
		return -days

	#sim_obj.simulate_day(1)
	sim_obj.simulate_days(50)
	#sim_obj.simulate_day(7)

	#sim_obj.simulate_till_end(reward_fn)
	sim_obj.grid.animate(False,color_list,0.3)
	sim_obj.grid.plot_time_series()
