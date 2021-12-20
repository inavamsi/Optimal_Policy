import numpy as np
import argparse
from environment import game_env
from agent import get_agent
from utils import plot_learning_curve

def parse_args():
	arg_parser = argparse.ArgumentParser()

	# input argument options
	arg_parser.add_argument("-gs", "--grid_size", dest="grid_size", type=int, default=12, help="Grid size S x S. Default S = 12")
	arg_parser.add_argument("-vs", "--vax_size", dest="vax_size", type=int, default=4, help="Vaccination size V x V. Default V = 4")
	arg_parser.add_argument("-net", "--network", dest="network", type=str, default="ConvDQN", help="Network to use. Options = LinearDQN, SimpleConvDQN, ConvDQN, DoubleDQN, DuelingDQN, DuelingDoubleDQN. Default = ConvDQN")
	arg_parser.add_argument("-me", "--max_epd", dest="max_epd", type=int, default=10000, help="Max Episodes to run. Default = 10000")
	arg_parser.add_argument("-lr", "--learning_rate", dest="learning_rate", type=float, default=3e-2, help="Learning Rate. Default = 3e-2")
	arg_parser.add_argument("-g", "--gamma", dest="gamma", type=float, default=0.95, help="Gamma value. Default = 0.95")
	arg_parser.add_argument("-bs", "--batch_size", dest="batch_size", type=int, default=16, help="Batch Size. Default = 16")
	arg_parser.add_argument("-buf", "--buffer", dest="max_buffer_size", type=int, default= 10000, help= "Max buffer size. Default = 10000")
	arg_parser.add_argument("-emx", "--eps_max", dest="eps_max", type=float, default=1.0, help="Maximum Epsilon. Default = 1.0")
	arg_parser.add_argument("-emn", "--eps_min", dest="eps_min", type=float, default=0.01, help="Minimum Epsilon. Default = 0.01")
	arg_parser.add_argument("-ed", "--eps_dec", dest="eps_dec", type=float, default=0.0001, help="Epsilon Decrement. Default = 0.0001")
	arg_parser.add_argument("-us", "--update_steps", dest="update_steps", type=int, default=200, help="Target Network update steps for ConvDQN. Default = 200")
	args = arg_parser.parse_args()
	return args

def other_defaults():
	individual_types=['Susceptible','Infected','Immune','Vaccinated']
	initial_types_pop = {'Susceptible':0.98, 'Infected':0.02, 'Immune':0.0, 'Vaccinated':0.0}
	color_list=['black','red','white','blue']

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

	transmission_prob={}
	for t in individual_types:
		transmission_prob[t]={}

	for t1 in individual_types:
		for t2 in individual_types:
			transmission_prob[t1][t2]=p_standard(0)

	transmission_prob['Susceptible']['Infected']= p_infection
	transmission_prob['Infected']['Immune']= p_standard(0.2)
	transmission_prob['Immune']['Susceptible']= p_standard(0)

	return individual_types, initial_types_pop, transmission_prob, color_list

if __name__ == "__main__":

	# Parse Arguments
	args = parse_args()

	# Other Defaults
	individual_types, initial_types_pop, transmission_prob, color_list = other_defaults()

	# RL Environment and Agent
	env = game_env(args.grid_size, individual_types, initial_types_pop, transmission_prob, color_list, args.vax_size)
	agent = get_agent(env, args)

	# RL run
	episode_rewards = []
	eps_history = []

	for episode in range(args.max_epd):
		state = env.reset()
		episode_reward = 0
		done = False
		step = 0

		while not done:
			action = agent.get_action(state)
			next_state, reward, done, _ = env.step(action)
			agent.learn(state, action, reward, next_state, done, args.batch_size)
			episode_reward += reward
			state = next_state

		episode_rewards.append(episode_reward)
		eps_history.append(agent.eps)

		if (episode+1) % 100 == 0:
			avg_score = np.mean(episode_rewards[-100:])
			print("Episode " + str(episode+1) + ": " + str(avg_score))

	# Plots
	inp = 1
	while inp == 1:
		plot_learning_curve(episode_rewards, eps_history)
		env.env_plot()
		try:
			inp = int(input("Rerun plots? (1 for Yes) : "))
		except:
			break
