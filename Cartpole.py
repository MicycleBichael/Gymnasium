'''
Trains an algorithm on the Cart Pole environment using deep Q-Learning
'''
import gymnasium as gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

# Initializes environments
env = gym.make("CartPole-v1")
env2 = gym.make("CartPole-v1", render_mode="human")

# Initializes model
inputs = tf.keras.Input(shape=(4,))
x = tf.keras.layers.Dense(32, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(2, activation=tf.nn.relu)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

EPSILON = 0.6  # Chance of taking a random action
DISCOUNT = 0.12
LEARNING_RATE = 0.3
BATCH_SIZE = 1000  # Number of episodes to train on at once
EP_COUNT = 200000  # Number of episodes
DECAY_MULT = (0.1/EPSILON)**(1/EP_COUNT)  # Decreases EPSILON to 0.001 over EP_COUNT episodes
MAX_ITERATIONS = 100  # maximum batch iterations before quitting
batch_experiences = []
batch_validation = []  # validation set
optimizer = tf.optimizers.SGD(learning_rate=LEARNING_RATE)
mse_loss = tf.keras.losses.MeanSquaredError()
best_model = None


def get_next_action(state_in):
    '''
    Obtains next action through epsilon greedy algorithm
    '''
    if np.random.random() > EPSILON:
        state_in = tf.convert_to_tensor(state_in.reshape((1, 4)))
        return tf.argmax(model(state_in), axis=-1).numpy()[0]
    else:
        return np.random.randint(2)


reward_arr = []  # Array of rewards
loss_arr = []  # Array of losses
last_avg = 0  # last average of training sesh
iterations = 0  # batch iterations without improvement
episodes = 0  # episodes over last training seshs
with tf.device('/GPU:0'):
    for _ in range(EP_COUNT):
        terminated = False
        truncated = False
        observation, info = env.reset()
        reward_sum = 0
        loss_sum = 0
        while not terminated and not truncated:
            state = observation  # Stores previous observation
            if _ % 2 == 0:
                actionIndex = get_next_action(observation)
            else:
                temp_state = tf.convert_to_tensor(observation.reshape((1, 4)))
                actionIndex = tf.argmax(model(temp_state), axis=-1).numpy()[0]
            observation, reward, terminated, truncated, info = env.step(actionIndex)
            experience = [state, observation, reward, actionIndex]
            if _ % 2 == 0:
                batch_experiences.append(experience)  # Stores results in a batch array
            else:
                batch_validation.append(experience)
            if len(batch_experiences) == BATCH_SIZE:
                mean = sum([exp[2] for exp in batch_validation])/episodes
                if mean > last_avg:
                    last_avg = mean
                    best_model = model
                else:
                    iterations += 1
                episodes = 0
                print(f"\rEpisode: {_+1} || Training, please wait...", end="")
                inputs = np.array([exp[0] for exp in batch_experiences])
                targets = []
                for exp in batch_experiences:
                    temp = tf.convert_to_tensor(exp[1].reshape(1, 4))
                    targets.append(exp[2] + DISCOUNT*np.max(model(temp)))
                targets = np.array(targets).reshape(BATCH_SIZE, 1)
                with tf.GradientTape() as tape:
                    outputs = model(inputs)
                    loss = mse_loss(outputs, targets)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                batch_experiences = []
                batch_validation = []
                loss_sum += loss.numpy()
            reward_sum += reward
        episodes += 1
        # EPSILON *= DECAY_MULT
        if iterations >= MAX_ITERATIONS:
            break
        loss_arr.append(loss_sum)
        reward_arr.append(reward_sum)
        print(f"\rEpisode: {_+1} || Running! ^_^            ", end="")
print(f"\nBest Average Score: {last_avg}")
plt.subplot(121)
plt.plot(reward_arr, 'r-', label='score', linewidth=1)
plt.xlabel('Episode')
plt.legend()
plt.subplot(122)
plt.plot(loss_arr, 'b-', label='loss', linewidth=1)
plt.xlabel('Episode')
plt.legend()
plt.show()

# Visualization
terminated = False
observation, info = env2.reset()
observation = tf.convert_to_tensor(observation.reshape((1, 4)))
while not terminated:
    actionIndex = tf.argmax(best_model(observation), axis=-1).numpy()[0]
    observation, reward, terminated, truncated, info = env2.step(actionIndex)
    observation = tf.convert_to_tensor(observation.reshape((1, 4)))
