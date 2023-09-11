import numpy as np
import tensorflow as tf
import collections
import statistics
import tqdm
import matplotlib.pyplot as plt
import os
import cv2
import atexit
import shutil
import keyboard

from typing import List, Tuple
from PIL import Image

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

seed = 69
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.96,
    decay_steps=60000,
    decay_rate=0.9998478666
)

num_hidden_units = 256
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
dir_name = "mnistdiscriminationCAPPED"
SAVE_PATH = f"C:/Users/potot/Desktop/code/Research/Gymnasium/Saved Models/{dir_name}{num_hidden_units}/"
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_traintemp = []
y_traintemp = []
x_testtemp = []
y_testtemp = []
num_cap = 2
for i, train in enumerate(x_train):
    if y_train[i] < num_cap and len(train) > 1:
        x_traintemp.append(np.array(train,dtype='int32'))
        y_traintemp.append(y_train[i])
for i, test in enumerate(x_test):
    if y_test[i] < num_cap and len(test) > 1:
        x_testtemp.append(np.array(test,dtype='int32'))
        y_testtemp.append(y_test[i])
x_test = np.array(x_testtemp)
x_train = np.array(x_traintemp)
y_test = np.array(y_testtemp)
y_train = np.array(y_traintemp)


class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""
    def __init__(
        self,
        num_actions: int,
            num_hidden_units: int,
            image_shape: Tuple[int,int]):
        """Initialize."""
        super().__init__()

        self.common = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=image_shape),
            tf.keras.layers.Dense(num_hidden_units, activation='relu')
        ])
        self.actor = tf.keras.layers.Dense(num_actions)
        self.critic = tf.keras.layers.Dense(1)

        # Add a flatten layer to the critic output to remove the spatial dimensions
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        actor_output = self.actor(x)
        critic_output = self.critic(x)
        return actor_output, critic_output
    

def save(model: tf.keras.Model):
    dir_list = os.listdir(SAVE_PATH)
    num_saves = 8
    if len(dir_list) >= num_saves:
        for folder in dir_list:
            filepath = os.path.join(SAVE_PATH,folder)
            if str(folder) == "1":
                shutil.rmtree(os.path.join(SAVE_PATH, folder))
                continue
            os.rename(filepath,f"{SAVE_PATH}{int(folder)-1}")
        new_path = os.path.join(SAVE_PATH,f"{num_saves}")
        os.makedirs(new_path)
        tf.keras.models.save_model(model, new_path)
    else:
        new_path = f"{SAVE_PATH}{len(dir_list)+1}"
        os.makedirs(new_path)
        tf.keras.models.save_model(model, new_path)
    return


num_actions = num_cap
image_shape = (28,28,1)
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
if len(os.listdir(SAVE_PATH)) > 0:
    model = tf.keras.models.load_model(f"{SAVE_PATH}{os.listdir(SAVE_PATH)[-1]}")
    print(f"Loading model {os.listdir(SAVE_PATH)[-1]}...")
else:
    model = ActorCritic(num_actions, num_hidden_units, image_shape)




def cv_operations(state):
    """Groups computer vision operations done on observation state."""
    # Image.fromarray(state).save("Gymnasium/bruh.png")
    state = state.reshape((1,28,28,1))
    return state


def run_episode(
        state: tf.Tensor,
        model: tf.keras.Model, 
        label: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data."""

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    # Run the model and to get action probabilities and critic value
    action_logits_t, value = model(state)

    # Sample next action from the action probability distribution
    action = tf.random.categorical(action_logits_t, 1)[0, 0]
    action_probs_t = tf.nn.softmax(action_logits_t)

    # Store critic values
    values = values.write(0, tf.squeeze(value))

    # Store log probability of the action chosen
    action_probs = action_probs.write(0, action_probs_t[0, action])

    # Apply action to the environment to get next state and reward
    if action == label:
        reward = 100
    else:
        reward = -100

    # Store reward
    rewards = rewards.write(0, reward)

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards


def get_expected_return(
        rewards: tf.Tensor,
        gamma: float,
        standardize: bool = True) -> tf.Tensor:
    """Compute expected returns per timestep."""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) /
                   (tf.math.reduce_std(returns) + eps))

    return returns


def compute_loss(
        action_probs: tf.Tensor,
        values: tf.Tensor,
        returns: tf.Tensor) -> tf.Tensor:
    """Computes the combined Actor-Critic loss."""
    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss


@tf.function
def train_step(
        state: tf.Tensor,
        model: tf.keras.Model,
        label: int,
        optimizer: tf.keras.optimizers.Optimizer,
        gamma: float) -> tf.Tensor:
    """Runs a model training step."""
    
    with tf.GradientTape() as tape:

        # Run the model for one episode to collect training data
        action_probs, values, rewards = run_episode(
            state, model, label)

        # Calculate the expected returns
        returns = get_expected_return(rewards, gamma)

        # Convert training data to appropriate TF tensor shapes
        action_probs, values, returns = [
            tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

        # Calculate the loss values to update our network
        loss = compute_loss(action_probs, values, returns)

    # Compute the gradients from the loss
    grads = tape.gradient(loss, model.trainable_variables)
    clipped_gradients, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)

    # Apply the gradients to the model's parameters
    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward, loss

min_episodes_criterion = 400
max_episodes = len(x_train)

reward_threshold = 90
running_reward = 0
reward_arr = []
loss_arr = []

# The discount factor for future rewards
gamma = 0.99

# Keep the last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

t = tqdm.trange(max_episodes)
while running_reward < reward_threshold:
    for i in t:
        state = x_train[i]
        label = y_train[i]
        state = cv_operations(state)
        tf.config.run_functions_eagerly(True)
        episode_reward, loss = train_step(
            state, model, label, optimizer, gamma)
        episode_reward = int(episode_reward)
        episodes_reward.append(episode_reward)
        reward_arr.append(episode_reward)
        loss_arr.append(loss.numpy())
        running_reward = statistics.mean(episodes_reward)

        t.set_postfix(
            episode_reward=episode_reward, running_reward=running_reward)

        if i > 0 and i % 1000 == 0:
            save(model) 

        if running_reward > reward_threshold and i >= min_episodes_criterion:
            save(model)
            break   

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')