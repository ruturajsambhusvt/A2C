import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributions
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dim, fc1_size = 256, fc2_size = 256, name='Critic', checkpt_dir = 'tmp/a2c') -> None:
        super(CriticNetwork, self).__init__()

        self.beta = beta
        self.input_dim = input_dim
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.name = name
        self.checkpt_dir = checkpt_dir
        self.checkpt_file = os.path.join(self.checkpt_dir, name+'_a2c')
        
        # Define the layers
        self.fc1 = nn.Linear(*self.input_dim, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.fc3 = nn.Linear(self.fc2_size, 1)
        
        # Define the optimizer and device
        self.optimizer = optim.Adam(self.parameters(), lr=self.beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    
    def forward(self, state):
        value = self.fc1(state)
        value = F.relu(value)
        value = self.fc2(value)
        value = F.relu(value)
        value = self.fc3(value)
        
        return value
    
    def gradient_norm_clip(self,max_norm=0.5):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
    
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpt_file)
        
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpt_file))
        
        
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dim, action_dim, max_action, fc1_size = 256, fc2_size = 256, name='Actor', checkpt_dir = 'tmp/a2c') -> None:
        super(ActorNetwork,self).__init__()
        self.alpha = alpha
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.name = name
        self.checkpt_dir = checkpt_dir
        self.checkpt_file = os.path.join(self.checkpt_dir, name+'_a2c')
        
        #Define the layers
        self.fc1 = nn.Linear(*self.input_dim, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.mu = nn.Linear(self.fc2_size, self.action_dim)
        # self.sigma = nn.Linear(self.fc2_size, self.action_dim)
        logstds_param = nn.Parameter(torch.full((self.action_dim,),0.1))
        self.register_parameter('logstds',logstds_param)
        
        # Define the optimizer and device
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        
        mu = self.mu(prob)
        # sigma = self.sigma(prob)
        sigma = torch.clamp(self.logstds.exp(), min=1e-3, max=50)
        
        
        # sigma = torch.clamp(sigma, min=1e-6, max=1) # To avoid sigma = 0 
        # can try max = 1 to limit variance
    
        return torch.distributions.Normal(mu, sigma)
    
    def sample_action(self,state):
        
        policy = self.forward(torch.tensor(np.array(state), dtype=torch.float32).to(self.device))
        action = policy.sample().detach().cpu().numpy().flatten()
        
        # mu, sigma = self.forward(state)
        # policy = distributions.Normal(mu, sigma)
        # action_probs = policy.sample()
        
        #using tanh to limit the action to [-1,1] and multiply by max_action
        # action = torch.tanh(action_probs)*torch.tensor(self.max_action).to(self.device)
        # log_probs = policy.log_prob(action_probs).to(self.device)

        
        return action
    
    def gradient_norm_clip(self,max_norm=0.5):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
    
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpt_file)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpt_file))
        
        
        
        
        
        
