import numpy as np
#import tools
import gym

env = gym.make('FrozenLake-v1')

print(env.action_space)
print(env.observation_space.n)