import gym_super_mario_bros as gym

from nes_py.wrappers import JoypadSpace
from stable_baselines3.common import env_checker
from stable_baselines3 import A2C
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

envid = 'SuperMarioBros-1-1-v0'

env = gym.make(envid)
env = JoypadSpace(env,SIMPLE_MOVEMENT)

env_checker.check_env(env, warn=True, skip_render_check=True)

model = A2C("MlpPolicy", env, verbose=1,device="cuda")
model.learn(total_timesteps=300_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()pip