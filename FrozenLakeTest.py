import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# Actions: 0 = left, 1 = down, 2 = right, 3 = up

size = 4  # Map size, square
env = gym.make('FrozenLake-v1', desc=generate_random_map(size=size), is_slippery=True)
env2 = gym.make('FrozenLake-v1', desc=generate_random_map(size=size), is_slippery=True, render_mode="human")
q_values = np.zeros((size**2, 4))
epsilon = 0.5  # Chance of taking a random action
discount = 0.9
learningRate = 0.8
checkWait = 200  # Amount of episodes to wait before checking AI progress
epCount = 5000  # Number of episodes
decayMult = (0.001/epsilon)**(1/epCount)
mode = 'S'  # Switch between Q-Learning and SARSA-Learning


def getNextAction(currentIndex):
    if np.random.random() > epsilon:
        return np.argmax(q_values[currentIndex])
    else:
        return np.random.randint(4)


def updateQ(state, state2, reward, action):
    oldQValue = q_values[state, action]
    return oldQValue + learningRate*(reward + discount*(np.max(q_values[state2])) - oldQValue)


def updateS(state, state2, reward, action, action2):
    oldSARSA = q_values[state, action]
    return oldSARSA + learningRate*(reward + discount*(q_values[state2, action2]) - oldSARSA)


sumAverage = 0
for _ in range(epCount):
    terminated = False
    observation, info = env.reset()  # Initializes/resets environment, observation, and info values with base values
    rewardSum = 0  # Variable for holding sum of rewards over one episode
    while not terminated:
        oldIndex = observation  # Stores initial/current position in variables
        actionIndex = getNextAction(observation)
        observation, reward, terminated, truncated, info = env.step(actionIndex)
        rewardSum += reward
        if mode == 'Q':
            q_values[oldIndex, actionIndex] = updateQ(oldIndex, observation, reward, actionIndex)
        else:
            a2 = np.argmax(q_values[observation])
            q_values[oldIndex, actionIndex] = updateS(oldIndex, observation, reward, actionIndex, a2)
    sumAverage += rewardSum
    epsilon *= decayMult
    if (_+1) % checkWait == 0:
        print(f"Score: {sumAverage/checkWait}\nEpisode: {_+1}")
        print("=========================")
        sumAverage = 0

# visualization
terminated = False
observation, info = env2.reset()
while not terminated:
    oldIndex = observation  # stores initial position in variables
    actionIndex = np.argmax(q_values[observation])
    observation, reward, terminated, truncated, info = env2.step(actionIndex)
    oldQValue = q_values[oldIndex, actionIndex]
