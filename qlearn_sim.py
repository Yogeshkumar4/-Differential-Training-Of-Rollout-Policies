import gym
import numpy as np
import argparse
from environment import Environment
from qlearning import QLearningAgent
from utils import str2bool, ACTIONS
from joblib import Parallel, delayed

def simulate(side, instance, slip, obfuscate, randomseed, maxLength, gamma, num_episodes):
	env = Environment(side, instance, slip, obfuscate, randomseed, maxLength)	
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
	# print(episode_rewards[-100:])
	avg = np.mean(episode_rewards[-100:])
	pi=agent.getPi()
	print("Slip: "+str(slip)+" Avg: "+str(avg))
	env.printPolicy(pi)
	# print(episode_rewards[-1000:])
	# print("Mean episode reward: {}".format(np.mean(episode_rewards[-1000:])))
	return round(avg,4)

def qlearn_sim_run(args,slip_values,gamma):

	with Parallel(n_jobs = -1) as parallel:
		tempPi = parallel(delayed(simulate)(args.side, args.instance, s, args.obfuscate, args.randomseed, args.maxLength, gamma, args.numEpisodes) for s in slip_values)
		# pi = np.asarray(tempPi, dtype = 'int')

	return tempPi

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Implements the Environment.")
	parser.add_argument('-side', '--side', dest='side', type=int, default=8, help='Side length of the square grid')
	parser.add_argument('-i', '--instance', dest='instance', type=int, default=0, help='Instance number of the gridworld.')
	parser.add_argument('-ml', '--maxlength', dest='maxLength', type=int, default=1000, help='Maximum number of timesteps in an episode')
	parser.add_argument('-rs', '--randomseed', dest='randomseed', type=int, default=0, help='Seed for RNG.')
	parser.add_argument('-nobf', '--noobfuscate', dest='obfuscate', type=str2bool, nargs='?', const=False, default=False, help='Whether to obfuscate the states or not')
	parser.add_argument('-ne', '--numepisodes', dest='numEpisodes', type=int, default=10, help='Number of episodes to run')
	args = parser.parse_args()
	
	slip_values = [0.02,0.05,0.1,0.3,0.5,0.65,0.8,0.9]
	gamma = 0.95

	print(args)
	qlearn_sim_run(args,slip_values,gamma)