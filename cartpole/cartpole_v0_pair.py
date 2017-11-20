import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pyglet
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from common.gym_runner_pair import GymRunner
from common.Q_learning_pair import QLearningAgent


class CartPoleAgent(QLearningAgent):
    def __init__(self):
        super(CartPoleAgent, self).__init__(4, 2)

    def build_model(self):
        model = Sequential()
        model.add(Dense(12, activation='relu', input_dim=4))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(2))
        model.compile(Adam(lr=0.001), 'mse')

        # load the weights of the model if reusing previous training session
        # model.load_weights("models/cartpole-v0.h5")
        return model


if __name__ == "__main__":
    gym = GymRunner('CartPole-v0', 'gymresults/cartpole-v0')
    agent = CartPoleAgent()

    gym.train(agent, 1000)
    gym.run(agent, 500)

    agent.model.save_weights("models/cartpole-v0.h5", overwrite=True)
    gym.env.close()
    # gym.close_and_upload(os.environ['API_KEY'])
