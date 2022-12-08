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
import random

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, help= 'environment name', default='Pendulum-v1' ) #default='Pendulum-v1')
parser.add_argument('--algo', type=str, help='algorithm - REINFORCE or A2C',default='A2C') #default='A2C')
parser.add_argument('--mem_steps', type=int,  help='number of evaluation steps',default=32) #default=32,
parser.add_argument('--learning_steps', type=int,  help='total number of policy update steps',default=5000) #default=5000,
parser.add_argument('--learn', type=int,  help='learn or evaluate',default=1) #default=True,

args = parser.parse_args()

# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)

if __name__=='__main__':
    
    env = gym.make(args.env)
    path = '/Results/'+env.spec.id
    if not os.path.exists(os.getcwd()+path):
        os.makedirs(os.getcwd()+path)
    
    
    agent = Agent(env=env,alpha=0.0005,beta=0.005,layer1_critic=64,layer2_critic=64,layer1_actor=64,layer2_actor=64, gamma=0.95,mem_steps=args.mem_steps, algo=args.algo,max_grad_norm=0.5)
    
    figure_file = env.spec.id+agent.algo+'plot.png'
    title = env.spec.id
    
    best_score = env.reward_range[0]
       
    if args.learn:
        score_history = agent.learn(total_steps=args.learning_steps)
    
    else:
        agent.load_models()
        score_history = agent.evaluate(total_steps=5000)
    
    average_rewards = [np.mean(score_history[i:i+10]) for i in range(len(score_history))]
    
    plotLearning(score_history, title, figure_file, window=10)
        
        
