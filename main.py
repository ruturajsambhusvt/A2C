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
    env = gym.make('MinitaurBulletEnv-v0')
    # env = gym.make('Walker2DBulletEnv-v0')
    # env = gym.make('Pendulum-v1')
    
    
    agent = Agent(env=env,alpha=0.0005,beta=0.005,layer1_critic=64,layer2_critic=64,layer1_actor=64,layer2_actor=64, gamma=0.95,mem_steps=32, algo='A2C')
    path = '/Results/'+env.spec.id
    if not os.path.exists(os.getcwd()+path):
        os.makedirs(os.getcwd()+path)
    
    # np.random.seed(0)  #sensitive learning

    # writer = write_data('/Results/bulletInvertedPendulum')
    # writer = write_data('/Results/bulletDoubleInvertedPendulum')
    # writer = write_data('/Results/MinitaurBulletEnv-v0')
    writer = write_data(path+'/'+env.spec.id)
    
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    
    figure_file = env.spec.id+ 'plot.png'
    title = env.spec.id
    
    best_score = env.reward_range[0]
    load_checkpoint = False
    
    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')
       
    
    score_history = agent.learn(total_steps=5000)
    #5000 steps for gym pendulum
    #10000 steps for bullet env InvertedPendulum
    #10000 steps for bullet env InvertedDoublePendulum
    #5000 steps for bullet env MinitaurBulletEnv-v0
        
    average_rewards = [np.mean(score_history[i:i+10]) for i in range(len(score_history))]
    
    plotLearning(score_history, title, figure_file, window=10)
        
        
        
        
        
        
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
        
        
