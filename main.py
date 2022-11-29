import gym  
import pybullet, pybullet_envs
from Agent import Agent
from utils import plotLearning,write_data
import numpy as np
from gym import wrappers
import os
import torch
import torch.nn as nn
from torch import distributions
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

if __name__=='__main__':
    # env = gym.make('InvertedPendulumBulletEnv-v0')
    # env = gym.make('InvertedDoublePendulumBulletEnv-v0')
    # env = gym.make('CartPoleContinuousBulletEnv-v0')
    # env = gym.make('MinitaurBulletEnv-v0')
    # env = gym.make('Walker2DBulletEnv-v0')
    env = gym.make('Pendulum-v1')
    
    
    agent = Agent(env=env,alpha=0.0004,beta=0.004,layer1_critic=64,layer2_critic=64,layer1_actor=64,layer2_actor=64, gamma=0.99,mem_steps=32, algo='A2C')
    path = '/Results/'+env.spec.id
    if not os.path.exists(os.getcwd()+path):
        os.makedirs(os.getcwd()+path)
    
    np.random.seed(0)#sensitive learning

    # writer = write_data('/Results/bulletInvertedPendulum')
    # writer = write_data('/Results/bulletDoubleInvertedPendulum')
    # writer = write_data('/Results/MinitaurBulletEnv-v0')
    writer = write_data(path+'/'+env.spec.id)
    

    
    n_games = 100
    episode_length = agent.env._max_episode_steps
    total_steps = (n_games*episode_length)//agent.mem_steps
    
    
    # filename = 'double_inverted_pendulum.png'
    # filename = 'Walker2DBulletEnv.png'
    
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    
    
    figure_file = env.spec.id+ 'plot.png'
    
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    
    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')
       
    for i in range(n_games):
        agent.policy_eval()
        agent.learn()
        
        
    rewards = agent.episode_reward_store
    average_rewards = [np.mean(rewards[i:i+10]) for i in range(len(rewards))]
    y = np.arange(0, len(average_rewards))

    x = np.arange(0, len(rewards))

    fig, ax = plt.subplots()
    ax.plot(x, rewards)
    ax.plot(y, average_rewards)
        
        
        
        
        
        
    #     observation = env.reset()
    #     # env.render(mode='human')
    #     done = False
    #     score = 0
    #     for i in range(agent.mem_steps)
    #     while not done:
    #         action = agent.choose_action(observation)
    #         new_observation, reward, done, info = env.step(action)
    #         score += reward
    #         if not load_checkpoint:
    #             agent.learn(observation, action, reward, new_observation, done)
    #         observation = new_observation
    #     score_history.append(score)
    #     avg_score = np.mean(score_history[-100:])
        
    #     if avg_score > best_score:
    #         best_score = avg_score
    #         if not load_checkpoint:
    #             agent.save_models()
                
    #     print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)
        
    #     if not load_checkpoint:
    #             writer.write(i,score,avg_score)
        
        
    # if not load_checkpoint:
    #     x = [i+1 for i in range(n_games)]
    #     plotLearning(score_history, figure_file, window=100)
        
        
