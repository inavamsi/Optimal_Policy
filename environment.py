import numpy as np
import random
import copy
from policy import Vaccinate_block
from utils import animate, plot_time_series
from simulate import Simulate

class Cell():
	# This class refers to the Cell (Agent) that participates in the disease spread environnment.
	def __init__(self,individual_type,type_number,coordinates):
		self.individual_type=individual_type
		self.type_number=type_number
		self.policy_state={}
		self.policy_state['quarantined']=False
		self.policy_state['socially distanced']=False
		self.neighbours= []
		self.coordinates=coordinates

	def set_neighbours(self,neighbours):
		self.neighbours=neighbours

	def day(self):
	#When called simulates a day for cell.
		return None


class Grid():
	def __init__(self, grid, individual_types):
		self.grid_size=len(grid)
		self.initialise(individual_types)
		self.grid=grid
		self.update_timeseries()
		self.init_agent_grid()

	def init_agent_grid(self):
		self.agent_grid=[]
		for i in range(self.grid_size):
			self.agent_grid.append([])
			for j in range(self.grid_size):
				individual_type=self.number_to_type[self.grid[i][j]]
				agent=Cell(individual_type,self.grid[i][j],(i,j))
				self.agent_grid[i].append(agent)

		for i in range(self.grid_size):
			for j in range(self.grid_size):
				self.agent_grid[i][j].set_neighbours(self.nbr_agents(i,j))

	def initialise(self,individual_types):
		self.individual_types=individual_types      #List of Individual types
		self.day=0                                  #Scalar denoting current day
		self.no_types=len(individual_types)         #Scalar value denoting total number of individual types
		self.number_to_type={}            #Dictionary converting number in grid to individual type
		self.type_to_number={}            #Dictionary individual type to converting number in grid
		self.type_timeseries={}           #Dictionary of population time series of every type
		self.current_types_pop={}         #Dictionary of current population size of every type
		self.total_type_days={}           #Dictionary to store the total number of days a type has seen across population
		self.store=[]     #list to store history of grid


		for t in self.individual_types:
			self.type_timeseries[t]=[]

		for i in range(self.no_types):
			number=i
			cur_type=self.individual_types[i]
			self.type_to_number[cur_type]=number
			self.number_to_type[number]=cur_type

		for t in self.individual_types:
			self.current_types_pop[t]=-1 #uninitalised

	def randomly_intialize_grid(self,types_pop):
		self.grid=np.zeros((self.grid_size,self.grid_size))
		numbers=[]
		for i in range(len(types_pop)):
			for j in range(types_pop[i]):
				numbers.append(i)

		random.shuffle(numbers)

		for i in range(self.grid_size):
			for j in range(self.grid_size):
				self.grid[i][j]=numbers.pop(0)

	def update_timeseries(self):
		types_pop=np.zeros(self.no_types)
		for i in range(self.grid_size):
			for j in range(self.grid_size):
				types_pop[(int)(self.grid[i][j])]+=1
		for i in range(len(types_pop)):
			cur_type=self.number_to_type[i]
			self.type_timeseries[cur_type].append(types_pop[i])
			self.current_types_pop[cur_type]=types_pop[i]

		self.store.append(copy.deepcopy(self.grid))

	def convert_type(self, i,j, new_type):
		if i>=self.grid_size or i<0 or j>=self.grid_size or j<0:
			print("Error: Invalid grid coordinates!")
			return None
		old_type=self.number_to_type[self.grid[i][j]]
		self.grid[i][j]=self.type_to_number[new_type]

		self.agent_grid[i][j].individual_type=new_type

		self.current_types_pop[old_type]-=1
		self.current_types_pop[new_type]+=1

	def nbr_agents(self,i,j):
		nbr_agents=[]

		if i>0:
			nbr_agents.append(self.agent_grid[i-1][j])
		if j>0:
			nbr_agents.append(self.agent_grid[i][j-1])
		if i<self.grid_size-1:
			nbr_agents.append(self.agent_grid[i+1][j])
		if j<self.grid_size-1:
			nbr_agents.append(self.agent_grid[i][j+1])

		return nbr_agents

class game_env():
	def __init__(self,grid_size, individual_types, color_list, vaccination_size):
		self.individual_types=individual_types
		self.color_list=color_list
		self.vaccination_size = vaccination_size
		self.env_shape = (1, grid_size, grid_size)
		self.reset(grid_size)

	def sample_action(self):
		#return random.randint(0,self.sim_obj.policy.no_of_actions-1)
		return random.choice(self.sim_obj.policy.valid_actions)

	def reset(self, grid_size):
		def p_standard(p):
			def p_fn(day,global_state,a1,nbrs):  #probability of going from immune to susceptible.
				return p
			return p_fn

		def p_infection(day,global_state,my_agent,neighbour_agents):  # probability of infectiong neighbour
			p_inf=0.3
			p_not_inf=1
			for nbr_agent in neighbour_agents:
				if nbr_agent.individual_type in ['Infected','Asymptomatic'] and not nbr_agent.policy_state['quarantined']:
					p_not_inf*=(1-p_inf)

			return 1 - p_not_inf

		gridtable =np.zeros((grid_size,grid_size))
		gridtable[grid_size//3][grid_size//2]=1
		gridtable[grid_size-1][grid_size-2]=1
		gridtable[2][2]=1
		grid=Grid(gridtable,self.individual_types)
		policy=Vaccinate_block(grid, self.individual_types,self.vaccination_size,0)

		transmission_prob={}
		for t in self.individual_types:
			transmission_prob[t]={}

		for t1 in self.individual_types:
			for t2 in self.individual_types:
				transmission_prob[t1][t2]=p_standard(0)

		transmission_prob['Susceptible']['Infected']= p_infection
		transmission_prob['Infected']['Immune']= p_standard(0.2)
		transmission_prob['Immune']['Susceptible']= p_standard(0)

		self.sim_obj=sim_obj= Simulate(transmission_prob,self.individual_types,grid,policy)
		self.no_of_actions = policy.number_of_actions+1
		self.state=copy.deepcopy(sim_obj.grid.grid)
		return [grid.grid]

	def step(self,action):
		self.sim_obj.simulate_day(action)
		next_state=copy.deepcopy(self.sim_obj.grid.grid)

		# Reward Function
		done = False
		reward = - 1
		if self.sim_obj.grid.current_types_pop['Infected']==0:
			done = True
			reward = 0

		return [next_state], reward , done, None

	def env_plot(self):
		plot_time_series(self.sim_obj.grid)
		animate(self.sim_obj.grid,True,self.color_list,0.5)
