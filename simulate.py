from qlearn_sim import qlearn_sim_run
from qlearn_pair_sim import qlearn_pair_sim_run
import argparse,os
from utils import str2bool, ACTIONS
import matplotlib.pyplot as plt
from environment import Environment
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Implements the Environment.")
	parser.add_argument('-side', '--side', dest='side', type=int, default=8, help='Side length of the square grid')
	parser.add_argument('-i', '--instance', dest='instance', type=int, default=0, help='Instance number of the gridworld.')
	parser.add_argument('-ml', '--maxlength', dest='maxLength', type=int, default=1000, help='Maximum number of timesteps in an episode')
	parser.add_argument('-rs', '--randomseed', dest='randomseed', type=int, default=0, help='Seed for RNG.')
	parser.add_argument('-nobf', '--noobfuscate', dest='obfuscate', type=str2bool, nargs='?', const=False, default=False, help='Whether to obfuscate the states or not')
	parser.add_argument('-ne', '--numepisodes', dest='numEpisodes', type=int, default=10, help='Number of episodes to run')
	args = parser.parse_args()
	print(args)

	env0 = Environment(args.side, args.instance, 0, args.obfuscate, args.randomseed, args.maxLength)
	env0.printWorld()

	slip_values = [0]#,0.02,0.05,0.1,0.3,0.5,0.65,0.8,0.9]
	# slip_values = [0.6]
	gamma = 0.95
	print "Slip Values:",
	print(slip_values)

	# print("Running Q-learning...")
	# qlearn_avg = qlearn_sim_run(args,slip_values,gamma)
	# print(qlearn_avg)
	print("Running pairwise Q-learning...")
	qlearn_pair_avg = qlearn_pair_sim_run(args,slip_values,gamma)
	print(qlearn_pair_avg)

	plt.figure()
	plt.plot(slip_values,qlearn_avg,slip_values,qlearn_pair_avg)
	plt.title('Experiment \n Average reward vs Slip')
	plt.xlabel('Slip')
	plt.ylabel('Average episodic reward')
	plt.legend(('Q-learning benchmark','Pairwise Q-learning'))
	plt.show()