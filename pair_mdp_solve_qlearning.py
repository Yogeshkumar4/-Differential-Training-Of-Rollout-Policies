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
            actionMeans[i] += (r1 - r2) + gamma * V[ns1][ns2]
    actionMeans /= averageOver
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

def evaluatePi(pi, env, num=1000):
    rewards = 0
    env = copy.deepcopy(env)
    for i in range(num):
        s = env.reset()
        done = 'continue'
        reward = 0
        while done == 'continue':
            s1, r, done = env.step(ACTIONS[pi[s]])
            s = s1
            reward += r
        rewards += reward
    return rewards/num


class QLEnvAgent:

    def __init__(self, env, gamma, lr, epsilon=0.5):
        self.env = copy.deepcopy(env)
        self.gamma = gamma
        self.numActions = len(ACTIONS)
        self.Q = np.zeros((env.states, env.states, self.numActions, self.numActions))
        self.__initparams__()
        self.alpha = lr
        self.episode = 0

    def getAction(self):
        s = self.curr_state
        noise = np.random.randn(self.numActions, self.numActions)/(self.episode + 1)
        self.action = np.unravel_index(np.argmax(self.Q[s[0]][s[1]] + noise), noise.shape)
        return self.action

    def observe(self, newState, reward, done):
        s, a = self.curr_state, self.action
        alpha, gamma = self.alpha, self.gamma
        self.Q[s[0]][s[1]][a[0]][a[1]] += alpha * (reward + gamma* np.max(self.Q[newState[0]][[newState[1]]]) -
            self.Q[s[0]][s[1]][a[0]][a[1]])
        self.curr_state = newState
        if done:
            self.__initparams__()
            self.episode += 1

    def __initparams__(self):
        self.curr_state = env.reset()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Implements the Environment.")
    parser.add_argument('-side', '--side', dest='side', type=int, default=8, help='Side length of the square grid')
    parser.add_argument('-i', '--instance', dest='instance', type=int, default=0, help='Instance number of the gridworld.')
    parser.add_argument('-slip', '--slip', dest='slip', type=float, default=0.8, help='How likely is it for the agent to slip')
    parser.add_argument('-ml', '--maxlength', dest='maxLength', type=int, default=1000, help='Maximum number of timesteps in an episode')
    parser.add_argument('-rs', '--randomseed', dest='randomseed', type=int, default=0, help='Seed for RNG.')
    parser.add_argument('-nobf', '--noobfuscate', dest='obfuscate', type=str2bool, nargs='?', const=False, default=False, help='Whether to obfuscate the states or not')
    parser.add_argument('-ne', '--numepisodes', dest='numEpisodes', type=int, default=1600, help='Number of episodes to run')
    args = parser.parse_args()

    env0 = Environment(args.side, args.instance, args.slip, args.obfuscate, args.randomseed, args.maxLength)
    env = pairEnv(env0)
    env0.printWorld()
    pi = np.random.randint(4, size=env.states)

    gamma = 0.95
    num_episodes = 20000
    agent = QLEnvAgent(env, gamma, lr=0.8)
    episode_rewards = np.zeros(num_episodes)
    for i in range(num_episodes):
        done = False
        episode_reward = 0
        while not done:
            action = agent.getAction() # Take action
            state, reward, done = env.step(action)
            agent.observe(state, reward, done)
            episode_reward += reward
        episode_rewards[i] = episode_reward
    print(episode_rewards[-1000:])
    print("Mean episode reward: {}".format(np.mean(episode_rewards[-100:])))

    Q = agent.Q
    V = np.amax(Q, axis=(2,3)).reshape(env0.numStates, env0.numStates)
    print("------------")
    print(V)
    print("------------")
    pi = updatePi(pi, env0, gamma, V)
    env0.printPolicy(pi)
    printPolicy(pi)
    print("Avg. Reward: {}".format(evaluatePi(pi, env0)))
    sys.stdout.flush()