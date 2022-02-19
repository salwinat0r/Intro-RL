import gym
from stable_baselines3 import PPO

env = gym.make('MountainCar-v0')
env.reset()

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000, reset_num_timesteps=False)

episodes = 5

for i in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        print(rewards, obs.shape)