
from __future__ import division
import numpy as np
import gym
from itertools import product
import cPickle as pickle
from collections import Counter

"""
0: left, 1: down, 3: up, 2 : right
SFFF
FHFH
FFFH
HFFG
"""
pi = [1, 2, 1, 0, 1, 1, 1, 0, 2, 2, 1, 3, 3, 2, 2, 3]

class pairEnv:

	def __init__(self):
		self.env1 = gym.make('FrozenLake-v0')
		self.env2 = gym.make('FrozenLake-v0')
		self.states = self.env1.observation_space.n

	def step(self, action):
		o1, r1, d1, _ = self.env1.step(action[0])
		o2, r2, d2, _ = self.env2.step(action[1])
		return ((o1, o2), r1 - r2, d1 or d2)

	def reset(self):
		return (self.env1.reset(), self.env2.reset())


states_info = pickle.load(open('data.pk', 'rb'))

num_episodes = 1000
env = pairEnv()
V = np.zeros((env.states, env.states))
alpha = 0.5
gamma = 0.95
for i in xrange(num_episodes):
	s = env.reset()
	done = False
	while not done:
		action = (pi[s[0]], pi[s[1]])
		s1, R, done = env.step(action)
		V[s[0]][s[1]] += alpha * (R + gamma * V[s1[0]][s1[1]] - V[s[0]][s[1]])
		s = s1


def get_diff(a1_list, a2_list):

sample_env = gym.make('FrozenLake-v0')
num_states = sample_env.observation_space.n
pi_new = np.zeros(num_states)
for s in xrange(num_states):
	for a, s1, r in states_info[s]:
		if a == pi[s]:
			s_a, r_s = s1, r
			break
	max_val, action = 0, pi[s]
	for a, s1, r in states_info[s]:
		val = r - r_s + V[s1][s_a]
		if val > max_val:
			action = a
			max_val = val
	pi_new[s] = action

# print(pi)
# print(pi_new)