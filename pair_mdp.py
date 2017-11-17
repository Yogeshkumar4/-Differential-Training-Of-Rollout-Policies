
from __future__ import division
import numpy as np

import copy
import argparse
from environment import Environment
from utils import str2bool, ACTIONS

class pairEnv:

    def __init__(self, env):
        self.env1 = env #gym.make('FrozenLake-v0')
        self.env2 = copy.deepcopy(env)
        self.states = self.env1.getnumStates()

    def step(self, action):
        o1, r1, d1 = self.env1.step(ACTIONS[action[0]])
        o2, r2, d2 = self.env2.step(ACTIONS[action[1]])
        return ((o1, o2), r1 - r2, d1 or d2)

    def reset(self):
        return (self.env1.reset(), self.env2.reset())

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Implements the Environment.")
    parser.add_argument('-side', '--side', dest='side', type=int, default=8, help='Side length of the square grid')
    parser.add_argument('-i', '--instance', dest='instance', type=int, default=0, help='Instance number of the gridworld.')
    parser.add_argument('-slip', '--slip', dest='slip', type=float, default=0.4, help='How likely is it for the agent to slip')
    parser.add_argument('-ml', '--maxlength', dest='maxLength', type=int, default=1000, help='Maximum number of timesteps in an episode')
    parser.add_argument('-rs', '--randomseed', dest='randomseed', type=int, default=0, help='Seed for RNG.')
    parser.add_argument('-nobf', '--noobfuscate', dest='obfuscate', type=str2bool, nargs='?', const=False, default=False, help='Whether to obfuscate the states or not')
    parser.add_argument('-ne', '--numepisodes', dest='numEpisodes', type=int, default=1600, help='Number of episodes to run')
    args = parser.parse_args()
    env0 = Environment(args.side, args.instance, args.slip, args.obfuscate, args.randomseed, args.maxLength)
    num_episodes = 1000
    env = pairEnv(env0)
    V = np.zeros((env.states, env.states))
    alpha = 0.5
    gamma = 0.95
    pi = np.random.randint(4, size=env.states)
    for i in xrange(num_episodes):
        s = env.reset()
        done = 'continue'
        while done == 'continue':
            action = (pi[s[0]], pi[s[1]])
            s1, R, done = env.step(action)
            V[s[0]][s[1]] += alpha * (R + gamma * V[s1[0]][s1[1]] - V[s[0]][s[1]])

    print(V)