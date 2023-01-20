import gymnasium as gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym.utils.play import play

env = gym.make("CartPole-v1")
env2 = gym.make("CartPole-v1",render_mode="human")
inputs = tf.keras.Input(shape=(4,))
x = tf.keras.layers.Dense(32, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(2, activation=tf.nn.relu)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
epsilon = 0.3 # Chance of taking a random action
discount = 0.12
learningRate = 0.3
batchSize = 1000 # Number of episodes to train on at once 
batchExperiences = []
epCount = 1000 # Number of episodes
decayMult = (0.001/epsilon)**(1/epCount)
disMult = (0.395/discount)**(2/epCount)
mode = 'S' # Switch between Q-Learning and SARSA-Learning
optimizer = tf.optimizers.SGD(learning_rate=learningRate)
mse_loss = tf.keras.losses.MeanSquaredError()

def getNextAction(state):
    if np.random.random() > epsilon:
        state = tf.convert_to_tensor(state.reshape((1,4)))
        return tf.argmax(model(state),axis=-1).numpy()[0]
    else:
        return np.random.randint(2)

rewardArr = [] # Array of rewards
lossArr = [] # Array of losses
for _ in range(epCount):
    terminated = False
    truncated = False
    observation, info = env.reset() # Initializes/resets environment, initializes observation and info values with base values
    rewSum = 0
    lossSum = 0
    while terminated == False and truncated == False:
        state = observation # Stores previous observation
        actionIndex = getNextAction(observation) # Uses epsilon-greedy method to choose next action
        observation, reward, terminated, truncated, info = env.step(actionIndex)
        batchExperiences.append([state,observation,reward,actionIndex]) # Stores results in a batch array
        if len(batchExperiences) == batchSize:
            print(f"\rEpisode: {_+1} || Training, please wait...",end="")
            inputs = np.array([exp[0] for exp in batchExperiences])
            targets = []
            for exp in batchExperiences:
                temp = tf.convert_to_tensor(exp[1].reshape(1,4))
                targets.append(exp[2] + discount*np.max(model(temp)))
            targets = np.array(targets).reshape(batchSize, 1)
            with tf.GradientTape() as tape:
                outputs = model(inputs)
                loss = mse_loss(outputs,targets)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            batchExperiences = []
            lossSum += loss.numpy()
        rewSum += reward    
    lossArr.append(lossSum)
    rewardArr.append(rewSum)
    #if _ <= epCount/2:
        #discount*=disMult
    print(f"\rEpisode: {_+1}",end="")
plt.subplot(121)
plt.plot(rewardArr,'r-',label='score',linewidth=1)
plt.xlabel('Episode')
plt.legend()
plt.subplot(122)
plt.plot(lossArr,'b-',label='loss',linewidth=1)
plt.xlabel('Episode')
plt.legend()
plt.show()
#visualization 
terminated = False
observation, info = env2.reset()
observation = tf.convert_to_tensor(observation.reshape((1,4)))
while(terminated == False):
    actionIndex = tf.argmax(model(observation),axis=-1).numpy()[0]
    observation, reward, terminated, truncated, info = env2.step(actionIndex)
    observation = tf.convert_to_tensor(observation.reshape((1,4)))
