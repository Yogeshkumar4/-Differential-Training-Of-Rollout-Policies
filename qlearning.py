import gym
import numpy as np
import argparse
from environment import Environment

num_episodes = 30000
ACTIONS = ['up', 'down', 'left', 'right']

class QLearningAgent:

    def __init__(self, env, gamma, lr, epsilon=0.5):
        self.env = env
        self.gamma = gamma
        self.numActions = len(ACTIONS)
        self.Q = np.zeros((env.getnumStates(), self.numActions))
        self.__initparams__()
        self.alpha = lr
        self.episode = 0

    def getAction(self):
        noise = np.random.randn(1, self.numActions)/(self.episode + 1)
        self.action = np.argmax(self.Q[self.curr_state] + noise)
        return ACTIONS[self.action]

    def observe(self, newState, reward, event):
        s, a = self.curr_state, self.action
        alpha, gamma = self.alpha, self.gamma
        self.Q[s][a] = self.Q[s][a] + alpha * (reward + gamma* np.max(self.Q[newState]) - self.Q[s][a])
        self.curr_state = newState
        if event != 'continue':
            self.__initparams__()
            self.episode += 1

    def __initparams__(self):
        self.curr_state = env.reset()


def str2bool(v):
    # https://stackoverflow.com/a/43357954/2570622
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

        parser = argparse.ArgumentParser(description="Implements the Environment.")
        parser.add_argument('-side', '--side', dest='side', type=int, default=8, help='Side length of the square grid')
        parser.add_argument('-i', '--instance', dest='instance', type=int, default=0, help='Instance number of the gridworld.')
        parser.add_argument('-slip', '--slip', dest='slip', type=float, default=0.4, help='How likely is it for the agent to slip')
        parser.add_argument('-ml', '--maxlength', dest='maxLength', type=int, default=1000, help='Maximum number of timesteps in an episode')
        parser.add_argument('-rs', '--randomseed', dest='randomseed', type=int, default=0, help='Seed for RNG.')
        parser.add_argument('-nobf', '--noobfuscate', dest='obfuscate', type=str2bool, nargs='?', const=False, default=True, help='Whether to obfuscate the states or not')
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
                if i == 99:
                    env.printWorld()
                    print("--------------------")
                state, reward, event = env.step(action)
                agent.observe(state, reward, event)
                episode_reward += reward
            episode_rewards[i] = episode_reward
        print(episode_rewards[-1000:])
        print("Mean episode reward: {}".format(np.mean(episode_rewards[-100:])))