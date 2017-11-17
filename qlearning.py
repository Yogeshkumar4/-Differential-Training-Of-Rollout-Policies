import gym
import numpy as np

num_episodes = 30000

class QLearningAgent:

    def __init__(self, env, gamma, lr, epsilon=0.5):
    	self.env = env
        self.gamma = gamma
        self.Q = np.zeros((env.observation_space.n,
            env.action_space.n))
        self.numActions = env.action_space.n
        self.__initparams__()
        # self.epsilon = epsilon
        self.alpha = lr
        self.episode = 0

    def getAction(self):
        # prob = np.random.random()
        # if prob < self.epsilon:
            # self.action = np.random.choice(self.numActions)
        # else:
        noise = np.random.randn(1, self.numActions)/(self.episode + 1)
        self.action = np.argmax(self.Q[self.curr_state] + noise)
        return self.action

    def observe(self, newState, reward, done):
        s, a = self.curr_state, self.action
        alpha, gamma = self.alpha, self.gamma
        self.Q[s][a] = self.Q[s][a] + alpha * (reward + gamma* np.max(self.Q[newState]) - self.Q[s][a])
        self.curr_state = newState
        if done:
            self.__initparams__()
            # self.epsilon *= 0.9
            self.episode += 1
        # else:
            # self.epsilon *= 0.999

    def __initparams__(self):
    	self.curr_state = env.reset()



if __name__ == '__main__':
        env = gym.make('FrozenLake-v0')
        gamma = 0.95
        num_episodes = 10000
        agent = QLearningAgent(env, gamma, lr=0.8)
        episode_rewards = np.zeros(num_episodes)
        for i in range(num_episodes):
            done = False
            episode_reward = 0
            while not done:
                # if i >= 999:
                #     env.render()
                action = agent.getAction() # Take action
                state, reward, done, _ = env.step(action)
                agent.observe(state, reward, done)
                episode_reward += reward
            episode_rewards[i] = episode_reward
        print(episode_rewards[-1000:])
        print("Mean episode reward: {}".format(np.mean(episode_rewards[-100:])))