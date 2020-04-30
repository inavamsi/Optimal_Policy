import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import random
import copy

class Agent():
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
    #When called simulates a day for agent.
    return None


class Grid():
  def __init__(self, grid, individual_types):
    self.grid_size=len(grid)
    self.initialise(individual_types)
    self.grid=grid
    self.update_timeseries()
    self.init_agent_grid()

  #def __init__(self, grid_size, individual_types, initial_types_pop):
  # self.grid_size=grid_size                    #Scalar denoting length and breadth of grid
  # self.initialise(individual_types)
  # self.randomly_intialize_grid(initial_types_pop)
  # self.update_timeseries()
  # self.state='Normal'
    

  def init_agent_grid(self):
    self.agent_grid=[]
    for i in range(self.grid_size):
      self.agent_grid.append([])
      for j in range(self.grid_size):
        individual_type=self.number_to_type[self.grid[i][j]]
        agent=Agent(individual_type,self.grid[i][j],(i,j))
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

  def neighbours(self, i, j):
    neighbour_type_list=np.zeros(self.no_types)
    if i>=self.grid_size or i<0 or j>=self.grid_size or j<0:
      print("Error: Invalid grid coordinates!")
      return None

    if i>0:
      nbr_no = self.grid[i-1][j]
      neighbour_type_list[(int)(nbr_no)]+=1
    if j>0:
      nbr_no = self.grid[i][j-1]
      neighbour_type_list[(int)(nbr_no)]+=1
    if i<self.grid_size-1:
      nbr_no = self.grid[i+1][j]
      neighbour_type_list[(int)(nbr_no)]+=1
    if j<self.grid_size-1:
      nbr_no = self.grid[i][j+1]
      neighbour_type_list[(int)(nbr_no)]+=1

    return neighbour_type_list

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

  def plot_time_series(self):
    for t in self.individual_types:
      plt.plot(self.type_timeseries[t])

    plt.title('Population timeline of different individual types')
    plt.legend(self.individual_types,loc='upper right', shadow=True)
    plt.show()
  
  def plot_grid(self,gridlines,color_list):
    data=self.grid
    n=self.grid_size
    # create discrete colormap
    cmap = colors.ListedColormap(color_list)
    bounds=[-0.5]
    for i in range(self.no_types):
    	bounds.append(bounds[i]+1)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm)

    # draw gridlines
    if(gridlines):
      ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
      ax.set_xticks(np.arange(-.5, n, 1));
      ax.set_yticks(np.arange(-.5, n, 1));

    plt.show()

  def animate(self, gridlines,color_list, time):
    for g in self.store:
      data=g
      n=self.grid_size
      # create discrete colormap
      cmap = colors.ListedColormap(color_list)
      bounds=[-0.5]
      for i in range(self.no_types):
        bounds.append(bounds[i]+1)
      norm = colors.BoundaryNorm(bounds, cmap.N)

      fig, ax = plt.subplots()
      ax.imshow(data, cmap=cmap, norm=norm)

      # draw gridlines
      if(gridlines):
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-.5, n, 1));
        ax.set_yticks(np.arange(-.5, n, 1));
      plt.pause(time)

    plt.pause(5)
    inp=input()
    for g in self.store:
      plt.close()



