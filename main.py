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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, help= 'environment name' ) #default='Pendulum-v1')
parser.add_argument('--algo', type=str, help='algorithm - REINFORCE or A2C') #default='A2C')
parser.add_argument('--mem_steps', type=int,  help='number of evaluation steps') #default=32,
parser.add_argument('--total_steps', type=int,  help='total number of policy update steps') #default=5000,

args = parser.parse_args()

if __name__=='__main__':
    
    
    
    # env = gym.make('InvertedPendulumBulletEnv-v0')
    # env = gym.make('InvertedDoublePendulumBulletEnv-v0')
    # env = gym.make('CartPoleContinuousBulletEnv-v0')
    # env = gym.make('MinitaurBulletEnv-v0')
    # env = gym.make('Walker2DBulletEnv-v0')
    # env = gym.make("MountainCarContinuous-v0")
    # env = gym.make('Pendulum-v1')
    env = gym.make(args.env)
    path = '/Results/'+env.spec.id
    if not os.path.exists(os.getcwd()+path):
        os.makedirs(os.getcwd()+path)
    
    
    agent = Agent(env=env,alpha=0.0005,beta=0.005,layer1_critic=64,layer2_critic=64,layer1_actor=64,layer2_actor=64, gamma=0.95,mem_steps=args.mem_steps, algo=args.algo,max_grad_norm=0.5)
    
    
    
    # np.random.seed(0)  #sensitive learning

    # writer = write_data('/Results/bulletInvertedPendulum')
    # writer = write_data('/Results/bulletDoubleInvertedPendulum')
    # writer = write_data('/Results/MinitaurBulletEnv-v0')
    # writer = write_data(path+'/'+env.spec.id)
    
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    
    figure_file = env.spec.id+agent.algo+'plot.png'
    title = env.spec.id
    
    best_score = env.reward_range[0]
    load_checkpoint = False
    
    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')
       
    
    score_history = agent.learn(total_steps=args.total_steps)
    #5000 steps for gym pendulum A2C and 10000 RF
    #10000 steps for bullet env InvertedPendulum and 20000 RF
    #2500 steps for bullet env InvertedDoublePendulum and 2500 for RF
    #5000 steps for bullet env MinitaurBulletEnv-v0
    #5000 steps for bullet env ContinuousCartPole and 5000 for RF
    # 20000 steps for bullet env MountainCarContinuous AND 15000 for RF
    #10000 steps for bullet env Walker2DBulletEnv-v0 and 7500 for RF
    
    average_rewards = [np.mean(score_history[i:i+10]) for i in range(len(score_history))]
    
    plotLearning(score_history, title, figure_file, window=10)
        
        
