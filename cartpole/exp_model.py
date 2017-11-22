import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pyglet
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Subtract, Lambda, Add
from keras.optimizers import Adam
import keras.backend as K
from keras.engine.topology import Layer
from keras import layers, optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2

import numpy as np
import argparse

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
        shape[-1] *= self.output_dim
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
        shape[-1] *= self.output_dim
        return tuple(shape)


class CartPoleAgent(QLearningAgent):
    def __init__(self, state_space, action_space, buffer_size):
        self.state_space = state_space
        self.action_space = action_space
        super(CartPoleAgent, self).__init__(self.state_space, self.action_space, buffer_size)

    def build_model(self):
        model = Sequential()
        model.add(Dense(12, activation='tanh', input_dim=self.state_space))
        model.add(Dense(12, activation='tanh'))
        model.add(Dense(self.action_space))
        model.compile(Adam(lr=0.001), 'mse')

        # load the weights of the model if reusing previous training session
        model.load_weights("models/cartpole-v0.h5")
        return model

    def createRegularizedModel(self, hiddenLayers=[12,12], activationType="relu", learningRate=0.001):
        bias = True
        dropout = 0
        regularizationFactor = 0.01
        model = Sequential()
        if len(hiddenLayers) == 0: 
            model.add(Dense(self.action_space, input_shape=(self.state_space,)))
            model.add(Activation("linear"))
        else :
            model.add(Dense(hiddenLayers[0], activation=activationType, input_shape=(self.state_space,)))
            '''
            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else :
                model.add(Activation(activationType))
            '''
            for index in range(1, len(hiddenLayers)):
                layerSize = hiddenLayers[index]
                model.add(Dense(layerSize, activation=activationType))
                '''
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else :
                    model.add(Activation(activationType))
                '''    
                if dropout > 0:
                    model.add(Dropout(dropout))
            model.add(Dense(self.action_space))
            model.add(Activation("linear"))
        optimizer = optimizers.Adam(lr=learningRate)
        model.compile(loss="mse", optimizer=optimizer)
        model.summary()
        return model    

    def create_pair_model(self):
       input_1 = Input(shape=(self.state_space,))
       input_2 = Input(shape=(self.state_space,))
       out_1 = self.model(input_1)
       out_2 = self.model(input_2)
       out_pair_1 = Repeat_Layer(self.action_space)(out_1)
       out_pair_2 = Tile_Layer(self.action_space)(out_2)
       diff = Subtract()([out_pair_1, out_pair_2])
       output = Lambda(lambda x: x)(diff)
       p_model = Model(inputs = [input_1, input_2], outputs = output)
       p_model.compile(Adam(lr=0.001), 'mse')
       return p_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implements the Environment.")
    parser.add_argument('-env', '--env_name', dest='env_name', type=str,
        default="CartPole-v0", help='Name of the environment')
    parser.add_argument('-train', '--train_steps', dest='train_steps', type=int,
            default = 1000, help="Number of training steps")
    parser.add_argument('-test', '--test_steps', dest='test_steps', type=int,
            default= 500, help="Number of test steps")
    parser.add_argument('-buffer', '--buffer_size', dest='buffer_size', type=int,
            default= 20000, help="Number of buffer size")
    parser.add_argument('--render', dest='render', action='store_true', help="Whether to render the environment")
    parser.add_argument('--save', dest='save', action='store_true', help="Save the model weights")
    parser.add_argument('--tiles', dest='tile_coding', action='store_true', help="Use tile coding or not")
    parser.set_defaults(render=False, save = True, tile_coding=False)
    args = parser.parse_args()

    dir_results = os.path.join('gymresults', args.env_name)
    gym = GymRunner(args.env_name, render= args.render, tile_coding=args.tile_coding)
    agent = CartPoleAgent(gym.state_space(), gym.action_space(), args.buffer_size)
    print(agent.pair_model.summary())
    gym.train(agent, args.train_steps)
    weights_path = os.path.join('models', args.env_name + '_train_steps_{}'.format(args.train_steps) + '.h5')
    if args.save:
      agent.model.save_weights(weights_path, overwrite=True)
    gym.run(agent, args.test_steps)
    gym.env.close()
