import pyglet
import gym
from gym import wrappers
from pair_mdp import pairEnv


class GymRunner:
    def __init__(self, env_id, monitor_dir, max_timesteps=100000):
        self.monitor_dir = monitor_dir
        self.max_timesteps = max_timesteps

        self.env = pairEnv(gym.make(env_id))
        self.env = wrappers.Monitor(self.env, monitor_dir, force=True)
        self.env_obv_shape = self.env.observation_space.shape[0]

    def calc_reward(self, state, action, gym_reward, next_state, done):
        return gym_reward

    def train(self, agent, num_episodes):
        self.run(agent, num_episodes, do_train=True)

    def convert_state(state):
        state[0] = state[0].reshape(1, self.env_obv_shape)
        state[1] = state[1].reshape(1, self.env_obv_shape)
        return state

    def run(self, agent, num_episodes, do_train=False):
        for episode in range(num_episodes):
            state = convert_state(self.env.reset())
            total_reward = 0

            for t in range(self.max_timesteps):
                action1 = agent.select_action(state, do_train)
                action2 = agent.select_action(state_prime, do_train)
                action = (action1, action2)

                # execute the selected action
                next_state, reward, done = self.env.step(action)
                next_state = convert_state(next_state)
                # reward = self.calc_reward(state, action, reward, next_state, done)

                # record the results of the step
                if do_train:
                    agent.record(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state
                if done:
                    break

            # train the agent based on a sample of past experiences
            if do_train:
                agent.replay()

            print("episode: {}/{} | score: {} | e: {:.3f}".format(
                episode + 1, num_episodes, total_reward, agent.epsilon))

    def close_and_upload(self, api_key):
        self.env.close()
        gym.upload(self.monitor_dir, api_key=api_key)
