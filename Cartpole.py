import gymnasium as gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym.utils.play import play

env = gym.make("CartPole-v1")
env2 = gym.make("CartPole-v1",render_mode="human")
inputs = tf.keras.Input(shape=(4,))
x = tf.keras.layers.Dense(64, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(2, activation=tf.nn.relu)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
epsilon = 0.5 # Chance of taking a random action
discount = 0.12
learningRate = 0.8
batchSize = 10 # Number of episodes to train on at once 
epCount = 1000 # Number of episodes
decayMult = (0.001/epsilon)**(1/epCount)
disMult = (0.395/discount)**(2/epCount)
mode = 'S' # Switch between Q-Learning and SARSA-Learning
optimizer = tf.optimizers.SGD(learning_rate=learningRate)
mse_loss = tf.keras.losses.MeanSquaredError()

def getNextAction(state):
    if np.random.random() > epsilon:
        return tf.argmax(model(state),axis=-1).numpy()[0]
    else:
        return np.random.randint(2)
def customLoss(state,state2,reward,action):
    oldQ = model(state)[0][action]
    newQ = reward + discount*(tf.reduce_max(model(state2),axis=-1))
    mse = mse_loss(tf.reshape(newQ,[1]),tf.reshape(oldQ,[1]))
    return tf.keras.backend.mean(mse)
'''def updateS(state,state2,reward,action,action2): 
    oldSARSA = q_values[state,action]
    return oldSARSA + learningRate*(reward + discount*(q_values[state2,action2]) - oldSARSA)'''

rewardArr = [] # Array of rewards
lossArr = [] # Array of losses
epArr = [] # Array of episode numbers
for _ in range(epCount):
    terminated = False
    truncated = False
    observation, info = env.reset() # Initializes/resets environment, initializes observation and info values with base values
    rewSum = 0
    lossSum = 0
    observation = tf.convert_to_tensor(observation.reshape((1,4)))
    while terminated == False and truncated == False:
        state = observation
        with tf.GradientTape() as tape:
            if _ < 900:
                actionIndex = getNextAction(observation)
            else:
                actionIndex = tf.argmax(model(state),axis=-1).numpy()[0]
            observation, reward, terminated, truncated, info = env.step(actionIndex)
            observation = tf.convert_to_tensor(observation.reshape((1,4)))
            loss = customLoss(state,observation,reward,actionIndex)
            gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        rewSum += reward
        lossSum += loss.numpy()
    lossArr.append(lossSum)
    rewardArr.append(rewSum)
    epArr.append(_+1)
    #if _ <= epCount/2:
        #discount*=disMult
    print(f"\rEpisode: {_+1}",end="")
    '''if (_+1) % checkWait == 0:
        print(f"Score: {rewardAvg/checkWait}\nLoss: {lossAvg/checkWait}\nEpisode: {_+1}")
        print( "=========================")
        rewardAvg = 0
        lossAvg = 0'''

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
