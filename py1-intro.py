import gym
from stable_baselines3 import a2c

env = gym.make('LunarLander-v2')

env.reset()

model = a2c.A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        obs, reward, done, info = env.step(model.predict(obs)[0])
        print(reward)
    

env.close()



