import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import random
import copy
import Grid
import Policy
import time

class Simulate():
	def __init__(self, p_infection, p_recovery, p_unimmunisation, individual_types,grid,policy):
		self.individual_types=individual_types
		self.grid=grid
		self.p_infection=p_infection
		self.p_recovery=p_recovery
		self.p_unimmunisation=p_unimmunisation
		self.day=0
		self.policy=policy
		self.total_infected_days=0

	def simulate_day(self,action_no):
		self.policy.do_action(self.grid,action_no)
		grid=self.grid
		new_grid=copy.deepcopy(grid)

		for i in range(grid.grid_size):
			for j in range(grid.grid_size):
				cur_type=grid.number_to_type[new_grid.grid[i][j]]
				if cur_type=='Infected':
					r=random.random()
					if r<self.p_recovery(self.day,self.grid.current_types_pop,grid.state):
						grid.convert_type(i,j,'Immune')

				elif cur_type=='Immune':
					r=random.random()
					if r<self.p_unimmunisation(self.day,self.grid.current_types_pop,grid.state):
						grid.convert_type(i,j,'Susceptible')

				elif cur_type=='Susceptible':
					neighbour_list=new_grid.neighbours(i,j)
					no_infected=neighbour_list[grid.type_to_number['Infected']]
					p_infection_from_nbrs=1 - (1-self.p_infection(self.day,self.grid.current_types_pop,grid.state))**no_infected
					r=random.random()
					if r<p_infection_from_nbrs:
						grid.convert_type(i,j,'Infected')

				else:
					print('Error: Invalid type!')
					return None
		self.day+=1
		grid.update_timeseries()
		self.total_infected_days+=self.grid.current_types_pop['Infected']
		return 

	def simulate_days(self,n):
		for i in range(n):
			self.simulate_day(-1)

	def simulate_till_end(self, reward_fn):
		no_infected=self.grid.current_types_pop['Infected']
		total_infected_days=0
		while(no_infected>0):
			action_no=random.choice(self.policy.valid_actions)
			self.simulate_day(action_no)
			no_infected=self.grid.current_types_pop['Infected']
			total_infected_days+=no_infected

		return reward_fn(self.day,total_infected_days)

def main():
	#Standard spread
	def p_infection(day,cur_type_pop,state):
		return 0.3
	def p_recovery(day,cur_type_pop,state):
		return 0.2
	def p_unimmunisation(day,cur_type_pop,state):
		return 0
	individual_types=['Susceptible','Infected','Immune']
	color_list=['white', 'black','red']
	gridtable =np.zeros((12,12))
	gridtable[5][6]=1
	gridtable[2][3]=1
	gridtable[7][2]=1
	grid=Grid.Grid(gridtable,individual_types)
	policy=Policy.Vaccinate_block(grid, individual_types,4,0)
	sim_obj= Simulate(p_infection, p_recovery, p_unimmunisation,individual_types,grid,policy)
	
	#sim_obj.simulate_day(1)
	#sim_obj.simulate_days(5)
	#sim_obj.simulate_day(7)

	#sim_obj.simulate_till_end()
	#sim_obj.grid.animate(False,color_list,0.1)
	#sim_obj.grid.plot_time_series()

for i in range(100):
 main()  













