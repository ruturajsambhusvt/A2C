import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributions
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dim, fc1_size = 256, fc2_size = 256, name='Critic', checkpt_dir = 'tmp/a2c') -> None:
        """This is the definition of Critic Network which represents the state valuues

        Args:
            beta (float): learning rate
            input_dim (np array): observation space
            fc1_size (int, optional): number of nuerons in layer 1. Defaults to 256.
            fc2_size (int, optional): number of neurons in layer 2. Defaults to 256.
            name (str, optional): name to be saved. Defaults to 'Critic'.
            checkpt_dir (str, optional): directory where check points are saved. Defaults to 'tmp/a2c'.
        """        
        super(CriticNetwork, self).__init__() #initialize the nn.Module class

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
        """forawrd pass of the network

        Args:
            state (np array): observations

        Returns:
            np array with single value: value of the state
        """        
        value = self.fc1(state)
        value = F.relu(value)
        value = self.fc2(value)
        value = F.relu(value)
        value = self.fc3(value)
        
        return value
    
    def gradient_norm_clip(self,max_norm=0.5):
        """gradient clipping to avoid exploding gradients

        Args:
            max_norm (float, optional): max value. Defaults to 0.5.
        """        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
    
    
    def save_checkpoint(self):
        """saving checkpoints
        """        
        torch.save(self.state_dict(), self.checkpt_file)
        
    def load_checkpoint(self):
        """loading checkpoints
        """        
        self.load_state_dict(torch.load(self.checkpt_file))
        
        
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dim, action_dim, max_action, fc1_size = 256, fc2_size = 256, name='Actor', checkpt_dir = 'tmp/a2c') -> None:
        """This is the definition of Actor Network which represents the policy

        Args:
            alpha (float): Actor learning rate
            input_dim (np array): observation space dim
            action_dim (np array): action space dimension
            max_action (float): max action value permitted
            fc1_size (int, optional): number of neurons in layer 1  . Defaults to 256.
            fc2_size (int, optional): number of nuerons in layer 2. Defaults to 256.
            name (str, optional): name to be saved. Defaults to 'Actor'.
            checkpt_dir (str, optional): directory where network is saved. Defaults to 'tmp/a2c'.
        """        
        
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
        #  logstd parameterization of the standard deviation
        logstds_param = nn.Parameter(torch.full((self.action_dim,),0.1))
        self.register_parameter('logstds',logstds_param)
        
        # Define the optimizer and device
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        """forward pass of the network

        Args:
            state (np array): observations

        Returns:
            torch distribution: normal distribution of the actions
        """        
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        mu = self.mu(prob)
        # clamp the mean to avoid numerical instability
        sigma = torch.clamp(self.logstds.exp(), min=1e-3, max=50)
        return torch.distributions.Normal(mu, sigma)
    
    def sample_action(self,state):
        """sample action from the policy

        Args:
            state (np array): observations

        Returns:
            np array: action
        """        
        
        policy = self.forward(torch.tensor(np.array(state), dtype=torch.float32).to(self.device))
        action = policy.sample().detach().cpu().numpy().flatten() # detach the action from the graph
       
        return action
    
    def gradient_norm_clip(self,max_norm=0.5):
        """gradient clipping to avoid exploding gradients

        Args:
            max_norm (float, optional): max value. Defaults to 0.5.
        """        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
    
    
    def save_checkpoint(self):
        """saving checkpoints
        """        
        torch.save(self.state_dict(), self.checkpt_file)
    
    def load_checkpoint(self):
        """loading checkpoints
        """
        self.load_state_dict(torch.load(self.checkpt_file))
        
        
class CriticNetworkLSTM(nn.Module):
    def __init__(self, beta, input_dim, fc1_size = 256, fc2_size = 256, hidden_size = 64, name='Critic', checkpt_dir = 'tmp/a2c') -> None:
        """This is the definition of Critic Network which represents the state valuues

        Args:
            beta (float): learning rate
            input_dim (np array): observation space
            fc1_size (int, optional): number of nuerons in layer 1. Defaults to 256.
            fc2_size (int, optional): number of neurons in layer 2. Defaults to 256.
            name (str, optional): name to be saved. Defaults to 'Critic'.
            checkpt_dir (str, optional): directory where check points are saved. Defaults to 'tmp/a2c'.
        """        
        super(CriticNetworkLSTM, self).__init__() #initialize the nn.Module class

        self.beta = beta
        self.input_dim = input_dim
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.hidden_size = hidden_size
        self.num_hidden_layer = 1
        self.name = name
        self.checkpt_dir = checkpt_dir
        self.checkpt_file = os.path.join(self.checkpt_dir, name+'_a2c')
        
        #Define lstm layer
        self.lstm = nn.LSTM
        
        # Define the layers
        self.fc1 = nn.Linear(*self.input_dim, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.fc3 = nn.Linear(self.fc2_size, 1)
        
        # Define the optimizer and device
        self.optimizer = optim.Adam(self.parameters(), lr=self.beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    
    def forward(self, state):
        """forawrd pass of the network

        Args:
            state (np array): observations

        Returns:
            np array with single value: value of the state
        """        
        value = self.fc1(state)
        value = F.relu(value)
        value = self.fc2(value)
        value = F.relu(value)
        value = self.fc3(value)
        
        return value
    
    def gradient_norm_clip(self,max_norm=0.5):
        """gradient clipping to avoid exploding gradients

        Args:
            max_norm (float, optional): max value. Defaults to 0.5.
        """        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
    
    
    def save_checkpoint(self):
        """saving checkpoints
        """        
        torch.save(self.state_dict(), self.checkpt_file)
        
    def load_checkpoint(self):
        """loading checkpoints
        """        
        self.load_state_dict(torch.load(self.checkpt_file))   
        


class ActorNetworkLSTM(nn.Module):
    def __init__(self, alpha, input_dim, action_dim, max_action, fc1_size = 256, fc2_size = 256, hidden_size = 64, name='Actor', checkpt_dir = 'tmp/a2c') -> None:
        """This is the definition of Actor Network which represents the policy

        Args:
            alpha (float): Actor learning rate
            input_dim (np array): observation space dim
            action_dim (np array): action space dimension
            max_action (float): max action value permitted
            fc1_size (int, optional): number of neurons in layer 1  . Defaults to 256.
            fc2_size (int, optional): number of nuerons in layer 2. Defaults to 256.
            name (str, optional): name to be saved. Defaults to 'Actor'.
            checkpt_dir (str, optional): directory where network is saved. Defaults to 'tmp/a2c'.
        """        
        
        super(ActorNetworkLSTM,self).__init__()
        self.alpha = alpha
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.hidden_size = hidden_size
        self.num_hidden_layer = 1
        self.name = name
        self.checkpt_dir = checkpt_dir
        self.checkpt_file = os.path.join(self.checkpt_dir, name+'_a2c')
        
        #Define the layers
        self.fc1 = nn.Linear(*self.input_dim, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.mu = nn.Linear(self.fc2_size, self.action_dim)
        #  logstd parameterization of the standard deviation
        logstds_param = nn.Parameter(torch.full((self.action_dim,),0.1))
        self.register_parameter('logstds',logstds_param)
        
        # Define the optimizer and device
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        """forward pass of the network

        Args:
            state (np array): observations

        Returns:
            torch distribution: normal distribution of the actions
        """        
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        mu = self.mu(prob)
        # clamp the mean to avoid numerical instability
        sigma = torch.clamp(self.logstds.exp(), min=1e-3, max=50)
        return torch.distributions.Normal(mu, sigma)
    
    def sample_action(self,state):
        """sample action from the policy

        Args:
            state (np array): observations

        Returns:
            np array: action
        """        
        
        policy = self.forward(torch.tensor(np.array(state), dtype=torch.float32).to(self.device))
        action = policy.sample().detach().cpu().numpy().flatten() # detach the action from the graph
       
        return action
    
    def gradient_norm_clip(self,max_norm=0.5):
        """gradient clipping to avoid exploding gradients

        Args:
            max_norm (float, optional): max value. Defaults to 0.5.
        """        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
    
    
    def save_checkpoint(self):
        """saving checkpoints
        """        
        torch.save(self.state_dict(), self.checkpt_file)
    
    def load_checkpoint(self):
        """loading checkpoints
        """
        self.load_state_dict(torch.load(self.checkpt_file)) 
