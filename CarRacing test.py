import gymnasium as gym
import numpy as np
import collections
import statistics
import tqdm
import matplotlib.pyplot as plt
import os
import cv2
from typing import List, Tuple
from gymnasium.utils.play import PlayPlot, play

env = gym.make("CarRacing-v2", continuous = False, render_mode="rgb_array")
mapping = {
    (ord('w'),):3,
    (ord('d'),):2,
    (ord('a'),):1,
    (ord('s'),):4,
    (): 0
}
def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    return [rew,]
plotter = PlayPlot(callback,150,["reward"])
play(env,keys_to_action=mapping,callback=plotter.callback)