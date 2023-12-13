'''
Trains an algorithm on the Cart Pole environment using deep Q-Learning
'''
import gymnasium as gym
import numpy as np
import tensorflow as tf
from keras import __version__
tf.keras.__version__ = __version__
import keras
import matplotlib.pyplot as plt
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# Initializes environments
env = gym.make("CartPole-v1")
states = env.observation_space.shape[0]
actions = env.action_space.n
env2 = gym.make("CartPole-v1", render_mode="human")

def build_model(states, actions):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(1,states)))
    model.add(keras.layers.Dense(24, activation = 'relu'))
    model.add(keras.layers.Dense(24, activation = 'relu'))
    model.add(keras.layers.Dense(actions, activation='linear'))
    return model
def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000,window_length = 1)
    dqn = DQNAgent(model = model,memory = memory,policy = policy,nb_actions = actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

model = build_model(states,actions)

dqn = build_agent(model,actions)
dqn.compile(keras.optimizers.Adam(lr=1e-3), metrics = ['mae'])
dqn.fit(env,nb_steps = 50000)