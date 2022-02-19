import gym
from stable_baselines3 import PPO
import os


models_dir = "models"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)


env = gym.make('BipedalWalker-v3') 
env.reset()

model = PPO('MlpPolicy', env, verbose=1)

TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{models_dir}/{TIMESTEPS*iters}")