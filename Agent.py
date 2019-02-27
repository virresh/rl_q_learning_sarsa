import numpy as np
import pickle
import math
import types

class Agent(object):
    def __init__(self, n, m, action_space, load_file=None, update_epsilon=None, update_alpha=None):
        self.rows = n
        self.cols = m
        if load_file:
            self.Q = pickle.load(load_file)
        else:
            self.Q = np.zeros((n*m, action_space.n))
        self.action_space = action_space
        self.action_taken = None
        self.previous_state = None
        self.epsilon = 0.8  # Exploration vs Exploitation
        self.alpha = 0.8    # Learning rate
        self.steps = 0
        self.gamma = 0.99    # Reward Discount factor
        if update_alpha:
            self.update_alpha = types.MethodType(update_alpha, self)
        if update_epsilon:
            self.update_epsilon = types.MethodType(update_epsilon, self)

    def convert_to_cell(self, state):
        # state is a list of two numbers, x and y index
        x,y = state
        cell_num = x * self.rows + y
        return int(cell_num)

    def act(self, state, policy='eps_greedy'):
        r = np.random.uniform(0,1)
        # print(state, self.convert_to_cell(state))
        state = self.convert_to_cell(state)
        if policy == 'eps_greedy':
            if r < self.epsilon:
                # exploration
                action = self.action_space.sample()
            else:
                # exploitation
                action = int(np.argmax(self.Q[state,:]))
        else:
            # Always exploit
            action = int(np.argmax(self.Q[state,:]))
        self.previous_state = state
        self.action_taken = action
        return action

    def learn(self, state, reward):
        # print(state)
        future = self.convert_to_cell(state)
        # print(state, future, np.amax(self.Q[future]))
        if self.previous_state  != None:
            # We have a previous state, so we can learn
            previous_val = self.Q[self.previous_state, self.action_taken]
            target = reward + self.gamma * np.max(self.Q[future, :])
            self.Q[self.previous_state, self.action_taken] += self.alpha * (target - previous_val)

    def end_episode(self):
        self.steps += 1
        self.update_alpha()
        self.update_epsilon()

    def update_alpha(self):
        self.alpha = max(self.alpha - 1/(self.steps+50), 0.3)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon - 1/(self.steps+50), 0.01)

    def save_table(self, file):
        pickle.dump(self.Q, file)

class  SARSAAgent(Agent):
    """ Implements SARSA algorithm"""
    def __init__(self, *args, **kwargs):
        super( SARSAAgent, self).__init__(*args, **kwargs)
        self.previous_state_2 = None
        self.action_taken_2 = None
        self.prev_rew = None

    def act(self, state, policy):
        self.previous_state_2 = self.previous_state
        self.action_taken_2 = self.action_taken
        act = super(SARSAAgent, self).act(state, policy)
        return act

    def learn(self, state, reward):
        if self.previous_state != None and self.previous_state_2 != None and self.prev_rew != None:
            future = self.previous_state
            future_act = self.action_taken
            current = self.previous_state_2
            current_act = self.action_taken_2

            prev_val = self.Q[current, current_act]
            new_val = self.prev_rew + self.gamma * self.Q[future, future_act]
            self.Q[current, current_act] += self.alpha * (new_val - prev_val)
        self.prev_rew = reward
