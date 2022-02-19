import gym
from stable_baselines3 import PPO

models_dir = "models"

env = gym.make('LunarLander-v2')  # continuous: LunarLanderContinuous-v2
env.reset()

model_path = f"{models_dir}/330000.zip"
model = PPO.load(model_path,env)

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        print(rewards, obs.shape)