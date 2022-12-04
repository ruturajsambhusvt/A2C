import gym

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env  = gym.make('Pendulum-v1')

model = A2C("MlpPolicy", env, verbose=1,learning_rate=0.005)
model.learn(total_timesteps=500000)
model.save("a2c_cartpole")

# del model # remove to demonstrate saving and loading

# model = A2C.load("a2c_cartpole")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()