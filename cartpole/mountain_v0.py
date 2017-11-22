import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pyglet
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from common.gym_runner import GymRunner
from common.q_learning_agent import QLearningAgent


class CartPoleAgent(QLearningAgent):
    def __init__(self):
        super(CartPoleAgent, self).__init__(2, 3, maxlen = 100000)

    def build_model(self):
        model = Sequential()
        model.add(Dense(30, activation='relu', input_dim=2))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(3))
        model.compile(Adam(lr=0.001), 'mse')

        # load the weights of the model if reusing previous training session
        # model.load_weights("models/cartpole-v0.h5")
        return model


if __name__ == "__main__":
    gym = GymRunner('MountainCar-v0', 'gymresults/mountaincar-v0', tile_coding=True)
    agent = CartPoleAgent()
    gym.train(agent, 3000)
    agent.model.save_weights("models/mountaincar-v0.h5", overwrite=True)
    gym.run(agent, 500)
    gym.env.close()
    # gym.close_and_upload(os.environ['API_KEY'])
