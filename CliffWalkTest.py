import gymnasium as gym
import numpy as np
from gym.utils.play import play
'''
mapping = {
    (ord('w'),):0,
    (ord('d'),):1,
    (ord('s'),):2,
    (ord('a'),):3
}
play(gym.make('CliffWalking-v0',render_mode="rgb_array"),keys_to_action=mapping)
'''
#Actions: 0 = Up, 1 = Right, 2 = Down, 3 = Left


env = gym.make('CliffWalking-v0')
env2 = gym.make('CliffWalking-v0',render_mode="human")
q_values = np.zeros((37,4))
epsilon = 0.5
discount = 0.9
learningRate = 0.9
checkWait = 10 #Amount of episodes to wait before checking AI progress

def getNextAction(currentIndex):
    if np.random.random() < epsilon:
        return np.argmax(q_values[currentIndex])
    else:
        return np.random.randint(4)
def updateQ(state,state2,reward,action):
    oldQValue = q_values[state,action]
    return oldQValue + learningRate*(reward + discount*(np.max(q_values[state2])) - oldQValue)
def updateS(state,state2,reward,action,action2):
    oldSARSA = q_values[state,action]
    return oldSARSA + learningRate*(reward + discount*(q_values[state2,action2]) - oldSARSA)

for _ in range(40):
    terminated = False
    observation, info = env.reset()
    rewardSum = 0
    while(terminated == False):
        oldIndex = observation #stores initial position in variables
        if _ % checkWait == 0:
            actionIndex = np.argmax(q_values[observation])
        else:
            actionIndex = getNextAction(observation)
        observation, reward, terminated, truncated, info = env.step(actionIndex)
        rewardSum += reward
        if observation > 36:
            break
        #q_values[oldIndex,actionIndex] = updateQ(oldIndex,observation,reward,actionIndex)
        q_values[oldIndex,actionIndex] = updateS(oldIndex,observation,reward,actionIndex,np.argmax(q_values[observation]))
    if _ % checkWait == 0:
        print(f"Score: {rewardSum}\nEpisode: {_}")

#visualization 
terminated = False
observation, info = env2.reset()
while(terminated == False):
    oldIndex = observation #stores initial position in variables
    actionIndex = np.argmax(q_values[observation])
    observation, reward, terminated, truncated, info = env2.step(actionIndex)
    oldQValue = q_values[oldIndex,actionIndex]
    if observation > 36:
            break
    newQValue = oldQValue + learningRate*(reward + discount*(np.max(q_values[observation])) - oldQValue)
    q_values[oldIndex,actionIndex] = newQValue
