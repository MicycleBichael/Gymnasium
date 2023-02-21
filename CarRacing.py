import gymnasium as gym
import numpy as np
import tensorflow as tf
import collections
import statistics
import tqdm
import matplotlib.pyplot as plt
import os
import cv2

from tensorflow.python.keras import layers
from typing import List, Tuple

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

env1 = gym.make("CarRacing-v2", continuous = False)
env2 = gym.make("CarRacing-v2", continuous = False, render_mode = "human")
env = env1

seed = 69
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

SAVE_PATH = "C:/Users/potot/Desktop/code/Research/Gymnasium/Saved Models/CarRacing"


class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""
    def __init__(
        self,
        num_actions: int,
            num_hidden_units: int):
        """Initialize."""
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

        # Add a flatten layer to the critic output to remove the spatial dimensions
        self.flatten = layers.Flatten()

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        actor_output = tf.reshape(self.actor(x), (-1, self.actor.units))
        critic_output = self.flatten(self.critic(x))
        return actor_output, critic_output


# Initializes model
num_actions = env.action_space.n  
num_hidden_units = 128
if len(os.listdir(SAVE_PATH)) > 0:
    model = tf.keras.models.load_model(SAVE_PATH)
    print("Loading model...")
else:
    model = ActorCritic(num_actions, num_hidden_units)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns state, reward and done flag given an action."""

    state, reward, done, truncated, info = env.step(action)
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    return (state.astype(np.float32),
            np.array(reward, np.int32),
            np.array(done, np.int32))


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action],
                             [tf.float32, tf.int32, tf.int32])


def run_episode(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data."""

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state
    reward_cache = collections.deque(maxlen=100)

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities and critic value
        action_logits_t, value = model(state)

        # Sample next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        # Store critic values
        values = values.write(t, tf.squeeze(value))

        # Store log probability of the action chosen
        action_probs = action_probs.write(t, action_probs_t[0, action])

        # Apply action to the environment to get next state and reward
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        # Store reward
        rewards = rewards.write(t, reward)
        reward_cache.append(int(reward))

        
        if len(reward_cache) >= 100 and statistics.mean(reward_cache) == 0:
            break
        if tf.cast(done, tf.bool):
            print(f"\nsteps: {t.numpy()}")
            break

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


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


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
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        gamma: float,
        max_steps_per_episode: int) -> tf.Tensor:
    """Runs a model training step."""
    
    with tf.GradientTape() as tape:

        # Run the model for one episode to collect training data
        action_probs, values, rewards = run_episode(
            initial_state, model, max_steps_per_episode)

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


def visualize(max_steps: int):
    initial_state, info = env2.reset()
    initial_state = cv2.cvtColor(initial_state,cv2.COLOR_RGB2GRAY)
    initial_state = tf.constant(initial_state, dtype=tf.float32)
    initial_state_shape = initial_state.shape
    state = initial_state
    for t in range(max_steps):
        state = tf.expand_dims(state, 0)
        action_logits_t, value = model(state)
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        state, reward, terminated, truncated, info = env2.step(action.numpy())
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = tf.constant(state, dtype=tf.float32)
        state.set_shape(initial_state_shape)
    env2.close()
    return

min_episodes_criterion = 100
max_episodes = 10000
max_steps_per_episode = 1000


reward_threshold = 475
running_reward = 0
reward_arr = []
loss_arr = []


# The discount factor for future rewards
gamma = 0.99

# Keep the last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

t = tqdm.trange(max_episodes)
for i in t:
    initial_state, info = env.reset()
    initial_state = cv2.cvtColor(initial_state,cv2.COLOR_RGB2GRAY)
    initial_state = tf.constant(initial_state, dtype=tf.float32)
    episode_reward, loss = train_step(
        initial_state, model, optimizer, gamma, max_steps_per_episode)

    episode_reward = int(episode_reward)
    episodes_reward.append(episode_reward)
    reward_arr.append(episode_reward)
    loss_arr.append(loss.numpy())
    running_reward = statistics.mean(episodes_reward)

    t.set_postfix(
        episode_reward=episode_reward, running_reward=running_reward)

    if i % 100 == 0:
        visualize(max_steps_per_episode)
        model.save(SAVE_PATH)

    if running_reward > reward_threshold and i >= min_episodes_criterion:
        model.save(SAVE_PATH)
        break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')


plt.subplot(121)
plt.plot(reward_arr, 'r-', label='score', linewidth=1)
plt.xlabel('Episode')
plt.legend()
plt.subplot(122)
plt.plot(loss_arr, 'b-', label='loss', linewidth=1)
plt.xlabel('Episode')
plt.legend()
plt.show()

visualize(max_steps_per_episode)