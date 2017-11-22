import abc
from collections import deque
import numpy as np
from keras.layers import Subtract, Input
from keras.models import Model
import random

class QLearningAgent(object):
    def __init__(self, state_size, action_size, buffer_size):
        self.state_size = state_size
        self.action_size = action_size

        # hyperparameters
        self.gamma = 0.99  # discount rate on future rewards
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.995  # the decay of epsilon after each training batch
        self.epsilon_min = 0.1  # the minimum exploration rate permissible
        self.batch_size = 32  # maximum size of the batches sampled from memory
        self.tile_coder = TileCoder()

        # agent state
        self.model = self.createRegularizedModel([10,10])
        self.pair_model = self.create_pair_model()
        # print(self.pair_model.summary())
        self.memory = deque(maxlen=buffer_size)

    @abc.abstractmethod
    def build_model(self):
        return None

    @abc.abstractmethod
    def create_pair_model(self):
        return None


    def select_action(self, state, do_train=True):
        if do_train and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state)[0])


    def record(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                v1 = np.amax(self.model.predict(next_state[0])[0])
                v2 = np.amax(self.model.predict(next_state[1])[0])
                target = (reward + self.gamma * (v1 - v2))

            target_f = self.pair_model.predict(state)
            action_index = action[0] * self.action_size + action[1]
            target_f[0][action_index] = target
            self.pair_model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
