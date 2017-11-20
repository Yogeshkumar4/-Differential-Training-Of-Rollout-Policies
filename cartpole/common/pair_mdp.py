import copy

class pairEnv:

    def __init__(self, env):
        self.env1 = copy.deepcopy(env)
        self.env2 = copy.deepcopy(env)
        self.observation_space = self.env1.observation_space

    def step(self, action):
        o1, r1, d1, _ = self.env1.step(action[0])
        o2, r2, d2, _ = self.env2.step(action[1])
        return ((o1, o2), r1 - r2, d1 or d2)

    def reset(self):
        return (self.env1.reset(), self.env2.reset())