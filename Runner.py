import gym
import gym_maze
import matplotlib.pyplot as plt
import numpy as np

from Agent import Agent, SARSAAgent

RENDER_MAZE = True

TOTAL_REWARD = []
TIME_TAKEN = []
EPSILON_VALS = []
ALPHA_VALS = []
STATE_TREND1 = []
STATE_TREND2 = []

def main(agent_class, train=False, load_file=None, policy=None,
		 EPOCHS=1000, ENV_STR='maze-sample-5x5-v0', MAX_T_MULT=100,
		 upd_eps = None,
		 upd_alpha = None,
		 initial_eps = None,
		 initial_alpha = None,
		 initial_gamma = None,
		 silent = False,
		 state1 = None,
		 state2 = None):
	"""
	Note : Put epochs = 1 for keeping things sane when using state1 and state2 variables
	"""
	global TOTAL_REWARD, TIME_TAKEN, EPSILON_VALS, ALPHA_VALS, STATE_TREND1, STATE_TREND2
	TOTAL_REWARD = []
	TIME_TAKEN = []
	EPSILON_VALS = []
	ALPHA_VALS = []
	STATE_TREND1 = []
	STATE_TREND2 = []

	env = gym.make(ENV_STR)
	# env = gym.make('maze-random-5x5-v0')
	N = int(env.observation_space.high[0]) + 1

	MAX_T = N*N*MAX_T_MULT
	IDEAL_T = N*N

	if load_file:
		with open(load_file, 'rb') as f:
			agent = agent_class(N, N, env.action_space, f, update_epsilon=upd_eps, update_alpha=upd_alpha)
	else:
		agent = agent_class(N, N, env.action_space, update_epsilon=upd_eps, update_alpha=upd_alpha)

	if initial_alpha != None:
		agent.alpha = initial_alpha
	if initial_eps != None:
		agent.epsilon = initial_eps
	if initial_gamma != None:
		agent.gamma = initial_gamma

	# Do for these many epochs
	for i in range(EPOCHS):
		state = env.reset()
		if RENDER_MAZE:
			env.render()
		done = False
		total_reward = 0
		for t in range(MAX_T):
			state, reward, done, _ = env.step(agent.act(state, policy))
			total_reward += reward
			if train:
				agent.learn(state, reward)
			if RENDER_MAZE:
				env.render()
			if done:
				break
		if not silent:
			print('Episode', i, 'ended in ', t, 'time steps with ', total_reward, 'reward')
		agent.end_episode()
		if state1 != None:
			STATE_TREND1.append(list(agent.Q[state1]))
		if state2 != None:
			STATE_TREND2.append(list(agent.Q[state2]))
		TOTAL_REWARD.append(total_reward)
		TIME_TAKEN.append(t)
		EPSILON_VALS.append(agent.epsilon)
		ALPHA_VALS.append(agent.alpha)

	if train:
		if load_file:
			with open(load_file, 'wb') as f:
				agent.save_table(f)
		else:
			with open('maze_agent.pkl', 'wb') as f:
				agent.save_table(f)

def plot_graph(title):
	fig, axs = plt.subplots(2, 2, constrained_layout=True)
	axs[0][0].plot(TIME_TAKEN)
	axs[0][0].set_title('Number of steps taken')
	axs[0][0].set_xlabel('Episodes')
	axs[0][0].set_ylabel('Steps')

	axs[0][1].plot(TOTAL_REWARD)
	axs[0][1].set_title('Total reward')
	axs[0][1].set_xlabel('Episodes')
	axs[0][1].set_ylabel('Total Reward')

	axs[1][0].plot(EPSILON_VALS)
	axs[1][0].set_title('Epsilon value')
	axs[1][0].set_xlabel('Episodes')
	axs[1][0].set_ylabel('Epsilon')
	
	axs[1][1].plot(ALPHA_VALS)
	axs[1][1].set_title('Alpha value')
	axs[1][1].set_xlabel('Episodes')
	axs[1][1].set_ylabel('Alpha')

	fig.suptitle(title)
	plt.show()

def find_converge_time(values):
	converged_val = min(values)
	for i in range(10, len(values)):
		found = True
		for k in values[i-5:i]:
			if k != converged_val:
				found = False
				break
		if found:
			return i-10
	return len(values)

if __name__ == '__main__':
	RENDER_MAZE = True

	def zero_eps(self):
		self.epsilon = 0

	def constant(self):
		pass

	## Part A
	# # Agent with constant alpha update
	# main(Agent, train=True, EPOCHS=50, MAX_T_MULT=8, upd_eps=zero_eps, upd_alpha=constant, initial_eps=0, policy=None)
	# plot_graph('Q-Learning, Constant Alpha, No explore, Constant Gamma')

	# # Agent with Episodic alpha decay and constant exploration
	# main(Agent, train=True, EPOCHS=100, MAX_T_MULT=8, upd_eps=constant, policy='eps_greedy')
	# plot_graph('Q-Learning, Episodic Alpha decay, Constant explore')

	# # Agent with Episodic alpha decay and Episodic exploration decay
	# main(Agent, train=True, EPOCHS=100, MAX_T_MULT=8, policy='eps_greedy')
	# plot_graph('Q-Learning, Episodic Alpha decay, Episodic explore decay')

	# # Agent with Episodic alpha decay
	# main(Agent, train=True, EPOCHS=100, MAX_T_MULT=8, upd_eps=zero_eps, initial_eps=0)
	# plot_graph('Q-Learning, Episodic Alpha decay, No explore')

	## Part B impact of hyperparameters
	# Effect of Gamma
	# gamma_plot_list = []
	# x = np.linspace(1, 0.0)
	# for gamma_value in x:
	# 	main(Agent, train=True, EPOCHS=50, MAX_T_MULT=8, policy=None, upd_eps=zero_eps, upd_alpha=constant, initial_gamma=gamma_value, silent=True)
	# 	gamma_plot_list.append(find_converge_time(TIME_TAKEN))
	# plt.figure()
	# plt.plot(x, gamma_plot_list)
	# plt.title('Convergence time vs Gamma')
	# plt.show()
	

	# Effect of Alpha
	# alpha_plot_list = []
	# x = np.linspace(0.9, 0.0)
	# for alpha_value in x:
	# 	main(Agent, train=True, EPOCHS=50, MAX_T_MULT=8, policy=None, upd_alpha=constant, initial_alpha=alpha_value, silent=True)
	# 	alpha_plot_list.append(find_converge_time(TIME_TAKEN))
	# plt.figure()
	# plt.plot(x, alpha_plot_list)
	# plt.title('Convergence time vs Alpha')
	# plt.show()

	## Part C
	# Observe probability of start state (i.e 0,0)
	# main(Agent, train=True, EPOCHS=50, MAX_T_MULT=8, upd_eps=zero_eps, state1=1, initial_eps=0)
	# arr = np.array(STATE_TREND1)
	# fig, axs = plt.subplots(2,2, constrained_layout=True)
	# axs[0][0].plot(arr[:,0])
	# axs[0][0].set_title('0')
	# axs[0][1].plot(arr[:,1])
	# axs[0][1].set_title('1')
	# axs[1][0].plot(arr[:,2])
	# axs[1][0].set_title('2')
	# axs[1][1].plot(arr[:,3])
	# axs[1][1].set_title('3')
	# print(arr.shape)
	# plt.show()
	# # RENDER_MAZE = True
	# # main(Agent, train=False, EPOCHS=100, MAX_T_MULT=8, load_file='maze_agent.pkl')

	## Part D
	# Effect of Espsilon
	# epsilon_plot_list = []
	# x = np.linspace(0.9, 0.0)
	# for epsilon_value in x:
	# 	main(Agent, train=True, EPOCHS=50, MAX_T_MULT=8, policy='eps_greedy', upd_eps=constant, initial_eps=epsilon_value, silent=True)
	# 	epsilon_plot_list.append(find_converge_time(TIME_TAKEN))
	# plt.figure()
	# plt.plot(x, epsilon_plot_list)
	# plt.title('Convergence time vs Epsilon')
	# plt.show()

	## Part E
	# Train on best parameters
	# RENDER_MAZE = False
	# main(Agent, train=True, EPOCHS=50, MAX_T_MULT=8, initial_alpha=0.8, initial_gamma=0.99, initial_eps=0.01, upd_eps=constant, upd_alpha=constant, policy='eps_greedy', silent=True)

	# # Test on best Parameters
	# # RENDER_MAZE = True
	# main(Agent, train=False, load_file='maze_agent.pkl', EPOCHS=50, MAX_T_MULT=8, initial_alpha=0.8, initial_gamma=0.99, initial_eps=0.01, upd_eps=constant, upd_alpha=constant, policy='eps_greedy')
	# plot_graph('Q-Learning, Best Parameters')

	# # Test intermediate states
	# main(Agent, train=True, EPOCHS=50, MAX_T_MULT=8, upd_eps=zero_eps, state1=0, state2=10, initial_eps=0)
	# arr = np.array(STATE_TREND1)
	# fig, axs = plt.subplots(2,2, constrained_layout=True)
	# axs[0][0].plot(arr[:,0])
	# axs[0][0].set_title('0')
	# axs[0][1].plot(arr[:,1])
	# axs[0][1].set_title('1')
	# axs[1][0].plot(arr[:,2])
	# axs[1][0].set_title('2')
	# axs[1][1].plot(arr[:,3])
	# axs[1][1].set_title('3')
	# print(arr.shape)

	# arr2 = np.array(STATE_TREND2)
	# fig, axs = plt.subplots(2,2, constrained_layout=True)
	# axs[0][0].plot(arr2[:,0])
	# axs[0][0].set_title('0')
	# axs[0][1].plot(arr2[:,1])
	# axs[0][1].set_title('1')
	# axs[1][0].plot(arr2[:,2])
	# axs[1][0].set_title('2')
	# axs[1][1].plot(arr2[:,3])
	# axs[1][1].set_title('3')
	# print(arr2.shape)
	# plt.show()

	# Question 4
	# Test on another maze env
	# main(Agent, ENV_STR='maze-random-5x5-v0', train=True, load_file='maze_agent.pkl')
	# plot_graph('Q-Learning, Best Parameters')

	# ## Bonus
	main(SARSAAgent, train=True, EPOCHS=50, MAX_T_MULT=8)
	plot_graph('SARSA Overview')
	
	# main(Agent, train=False, load_file='maze_agent.pkl')
