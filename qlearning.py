import gym
import numpy as np
import argparse
from environment import Environment
from utils import str2bool, ACTIONS
import random

num_episodes = 30000
ACTIONS = ['up', 'down', 'left', 'right']

class QLearningAgent:

    def __init__(self, env, gamma, lr, epsilon=0.5):
        self.env0 = env
        self.env = env
        self.gamma = gamma
        self.numActions = len(ACTIONS)
        self.Q = np.zeros((env.getnumStates(), self.numActions))
        self.__initparams__()
        self.alpha = lr
        self.epsilon = epsilon
        self.episode = 0

    def getAction(self):
        # noise = np.random.randn(1, self.numActions)/(self.episode + 1)
        if random.random()<self.epsilon:
            self.action = random.randint(0,self.numActions-1)
        else:
            self.action = np.argmax(self.Q[self.curr_state])
        return ACTIONS[self.action]

    def observe(self, newState, reward, event):
        s, a = self.curr_state, self.action
        alpha, gamma = self.alpha, self.gamma
        self.Q[s][a] = self.Q[s][a] + alpha * (reward + gamma* np.max(self.Q[newState]) - self.Q[s][a])
        self.curr_state = newState
        if event != 'continue':
            self.__initparams__()
            self.episode += 1

    def getPi(self):
        return np.argmax(self.Q,axis=1)

    def __initparams__(self):
        self.curr_state = self.env0.reset_start()

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
        env = Environment(args.side, args.instance, args.slip, args.obfuscate, args.randomseed, args.maxLength)
        gamma = 0.95
        num_episodes = 100
        agent = QLearningAgent(env, gamma, lr=0.8)
        episode_rewards = np.zeros(num_episodes)
        for i in range(num_episodes):
            event = 'continue'
            episode_reward = 0
            while event == 'continue':
                action = agent.getAction() # Take action
                state, reward, event = env.step(action)
                agent.observe(state, reward, event)
                episode_reward += reward
            episode_rewards[i] = episode_reward
        print(episode_rewards[-1000:])
        print("Mean episode reward: {}".format(np.mean(episode_rewards[-1000:])))