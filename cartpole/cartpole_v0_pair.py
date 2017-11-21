import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pyglet
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Subtract, Lambda, Add
from keras.optimizers import Adam
import keras.backend as K
from keras.engine.topology import Layer
from keras import layers

import numpy as np

from common.gym_runner_pair import GymRunner
from common.Q_learning_pair import QLearningAgent

class Repeat_Layer(layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Repeat_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Repeat_Layer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.repeat_elements(x, self.output_dim, axis=1)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        shape[-1] **= 2
        return tuple(shape)

class Tile_Layer(layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Tile_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Tile_Layer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.tile(x, [1, self.output_dim])

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        shape[-1] **= 2
        return tuple(shape)      


class CartPoleAgent(QLearningAgent):
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        super(CartPoleAgent, self).__init__(self.state_space, self.action_space)

    def build_model(self):
        model = Sequential()
        model.add(Dense(12, activation='relu', input_dim=self.state_space))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(self.action_space))
        model.compile(Adam(lr=0.001), 'mse')

        # load the weights of the model if reusing previous training session
        # model.load_weights("models/cartpole-v0.h5")
        return model

    def create_pair_model(self):
       input_1 = Input(shape=(self.state_space,))
       input_2 = Input(shape=(self.state_space,))
       out_1 = self.model(input_1)
       out_2 = self.model(input_2)
       out_pair_1 = Repeat_Layer(self.action_space)(out_1)
       out_pair_2 = Tile_Layer(self.action_space)(out_2)
       # out_pair_1 = K.repeat_elements(out_1, 2, axis=1)
       # out_pair_2 = K.tile(out_2, [1, 2])
       diff = Subtract()([out_pair_1, out_pair_2])
       # diff = out_pair_1 - out_pair_2
       # output = K.flatten(diff)
       output = Lambda(lambda x: x)(diff)
       p_model = Model(inputs = [input_1, input_2], outputs = output)
       p_model.compile(Adam(lr=0.001), 'mse')
       return p_model


if __name__ == "__main__":
    gym = GymRunner('MountainCar-v0', 'gymresults/cartpole-v0')
    agent = CartPoleAgent(gym.state_space(), gym.action_space())
    print(agent.pair_model.summary())
    gym.train(agent, 500)
    # print(agent.pair_model.get_weights())
    gym.run(agent, 500)

    agent.model.save_weights("models/cartpole-v0.h5", overwrite=True)
    gym.env.close()
    # gym.close_and_upload(os.environ['API_KEY'])
