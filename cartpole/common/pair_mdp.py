import copy
import gym
from gym import wrappers

class pairEnv:

    def __init__(self, env_id, render):
        env = gym.make(env_id)
        self.env1 = copy.deepcopy(env)
        self.env2 = copy.deepcopy(env)
        self.env_id = env_id
        self.observation_space = self.env1.observation_space
        self.action_space = self.env1.action_space
        self.reward = 0
        self.render = render
        self.alive_time = 0
        self.alive_time = 1


    def step(self, action, render = False):
        if render and self.render:
            self.env1.render()
        o1, r1, d1, _ = self.env1.step(action[0])
        o2, r2, d2, _ = self.env2.step(action[1])
        self.reward += r1
        pair_reward = self.modifiedReward(r1, d1) - self.modifiedReward(r2, d2)
        return ([o1, o2], pair_reward - 1, d1 or d2)

    def modifiedReward(self, r, d, end_reward = 50.0):
        if self.env_id == "CartPole-v0":
            return r  - self.alive_time * (d == True)
        elif self.env_id == "MountainCar-v0":
            return r + self.alive_time * (d == True)
        else:
            return r

    def reset(self):
        self.reward = 0
        self.alive_time = 0
        return [self.env1.reset(), self.env2.reset()]

    def close(self):
        self.env1.close()
        self.env2.close()
