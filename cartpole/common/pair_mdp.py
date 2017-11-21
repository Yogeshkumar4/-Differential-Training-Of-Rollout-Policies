import copy
from gym import wrappers

class pairEnv:

    def __init__(self, env):
        self.env1 = copy.deepcopy(env)
        # self.env1 = wrappers.Monitor(self.env1, 'gymresults/cartpole-v0-t', force=True)
        self.env2 = copy.deepcopy(env)
        self.observation_space = self.env1.observation_space
        self.action_space = self.env1.action_space
        self.reward = 0

    def step(self, action):
        self.env1.render()
        o1, r1, d1, _ = self.env1.step(action[0])
        o2, r2, d2, _ = self.env2.step(action[1])
        self.reward += r1
        return ([o1, o2], r1 - r2 +50*d1 - 50*d2, d1 or d2)

    def reset(self):
        self.reward = 0
        # self.env1.close()
        # self.env1 = wrappers.Monitor(self.env1, 'gymresults/cartpole-v0-t', force=True)
        return [self.env1.reset(), self.env2.reset()]