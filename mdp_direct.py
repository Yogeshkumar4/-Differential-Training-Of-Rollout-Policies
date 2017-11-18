
from __future__ import division
import numpy as np
import sys
import copy
import argparse
from environment import Environment
from utils import str2bool, ACTIONS
from joblib import Parallel, delayed

np.random.seed(42)

class pairEnv:

    def __init__(self, env):
        self.env1 = copy.deepcopy(env)
        self.env2 = copy.deepcopy(env)
        self.states = self.env1.getnumStates()

    def step(self, action):
        o1, r1, d1 = self.env1.step(ACTIONS[action[0]])
        o2, r2, d2 = self.env2.step(ACTIONS[action[1]])
        return ((o1, o2), r1 - r2, d1 != 'continue' or d2 != 'continue')

    def reset(self):
        return (self.env1.reset(), self.env2.reset())

def printPolicy(p):
    print([ACTIONS[i] for i in p])

def improveAction(state, policy_action, env0, gamma, V, averageOver = 1000):
    currentMax = -np.inf
    actionMeans = np.zeros(len(ACTIONS))
    for i, action in enumerate(ACTIONS):
        for j in xrange(averageOver):
            ns1, r1 = env0.sampleAction(state, action)
            ns2, r2 = env0.sampleAction(state, policy_action)
            actionMeans[i] += (r1 - r2) + gamma * V[ns1] - V[ns2]
    actionMeans /= averageOver
    # print(state, actionMeans)
    # print("------------------")
    if np.all(actionMeans == 0):
        bestAction = policy_action
    else:
        bestAction = np.argmax(actionMeans)
    return bestAction


def updatePi(pi, env0, gamma, V):
    with Parallel(n_jobs = -1) as parallel:
        tempPi = parallel(delayed(improveAction)(s, pi[s], env0, gamma, V) for
            s in xrange(env0.getnumStates()))
        pi = np.asarray(tempPi, dtype = 'int')
    return pi

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Implements the Environment.")
    parser.add_argument('-side', '--side', dest='side', type=int, default=3, help='Side length of the square grid')
    parser.add_argument('-i', '--instance', dest='instance', type=int, default=0, help='Instance number of the gridworld.')
    parser.add_argument('-slip', '--slip', dest='slip', type=float, default=0, help='How likely is it for the agent to slip')
    parser.add_argument('-ml', '--maxlength', dest='maxLength', type=int, default=1000, help='Maximum number of timesteps in an episode')
    parser.add_argument('-rs', '--randomseed', dest='randomseed', type=int, default=0, help='Seed for RNG.')
    parser.add_argument('-nobf', '--noobfuscate', dest='obfuscate', type=str2bool, nargs='?', const=False, default=False, help='Whether to obfuscate the states or not')
    parser.add_argument('-ne', '--numepisodes', dest='numEpisodes', type=int, default=1600, help='Number of episodes to run')
    args = parser.parse_args()

    env0 = Environment(args.side, args.instance, args.slip, args.obfuscate, args.randomseed, args.maxLength)
    env0.printWorld()
    pi = np.random.randint(4, size=env0.getnumStates())
    gamma = 0.95
    alpha, lamb = 0.2, 0
    for iteration in range(10):
        V = np.zeros(env0.getnumStates())
        pi = updatePi(pi, env0, gamma, V)
        num_episodes = 1000
        for _ in xrange(num_episodes):
            # e_trace = np.zeros_like(V)
            s = env0.reset()
            # print("Start {}:".format(s))
            states = []
            done = 'continue'
            while done == 'continue':
                states.append(s)
                action = pi[s]
                s1, R, done = env0.step(ACTIONS[action])
                delta = R + gamma * V[s1] - V[s]
                V[s] += alpha * delta
                s = s1
        print(V)
        pi = updatePi(pi, env0, gamma, V)
        printPolicy(pi)
        sys.stdout.flush()
