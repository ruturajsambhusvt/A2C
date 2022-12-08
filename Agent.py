import torch  
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributions
import numpy as np
from Network import ActorNetwork, CriticNetwork
from memory import Memory
from utils import write_data, plotLearning
import os

class Agent(object):
    """Agent is an object which performs evaluation and update of the policy

    Args:
        object (_type_): _description_
    """    
    def __init__(self,env,alpha,beta,layer1_critic=256,layer2_critic=256,layer1_actor=256,layer2_actor=256, gamma=0.99,mem_steps=32,algo='REINFORCE',max_grad_norm=0.5) -> None:
        """constructor for Agent class

        Args:
            env (gym env): gym environment
            alpha (float): actor learning rate
            beta (float): critic learning rate
            layer1_critic (int, optional): _description_. Defaults to 256.
            layer2_critic (int, optional): _description_. Defaults to 256.
            layer1_actor (int, optional): _description_. Defaults to 256.
            layer2_actor (int, optional): _description_. Defaults to 256.
            gamma (float, optional): discount factor. Defaults to 0.99.
            mem_steps (int, optional): number of evaluation steps. Defaults to 32.
            algo (str, optional): 'A2C' or 'REINFORCE'. Defaults to 'REINFORCE'.
            max_grad_norm (float, optional): gradient clipped value. Defaults to 0.5.
        """        
        self.env = env
        self.alpha = alpha
        self.beta = beta
        self.layer1_critic = layer1_critic
        self.layer2_critic = layer2_critic
        self.layer1_actor = layer1_actor
        self.layer2_actor = layer2_actor
        self.gamma = gamma
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape
        self.env_name = self.env.unwrapped.spec.id
        self.best_score = self.env.reward_range[0]
        
        self.mem_steps = mem_steps
        self.state = None
        self.done = True
        self.steps = 0
        self.episode_reward = 0
        self.episode_reward_store  = []
        self.algo = algo
        self.max_grad_norm = max_grad_norm
        self.writer = write_data(os.path.join(os.getcwd(),'/Results/'+env.spec.id+'/'+env.spec.id+'_'+algo))
        
        
        self.memory = Memory(self.algo)
        self.actor = ActorNetwork(alpha=alpha, input_dim=self.state_dim, action_dim= self.action_dim, max_action= self.env.action_space.high,fc1_size= self.layer1_actor, fc2_size= self.layer2_actor, name = 'Actor'+ self.env_name)
        self.critic = CriticNetwork(beta=beta, input_dim= self.state_dim, fc1_size= self.layer1_critic, fc2_size= self.layer2_critic, name =  'Critic' + self.env_name)
        
    def reset(self):
        """helper function to reset the environment object of the agent
        """        
        self.state = self.env.reset()
        self.done = False
        self.episode_reward = 0
        
    def policy_evalulation(self):
        """plays the game for a fixed number of steps and stores the experience in the memory
           idea from https://github.com/hermesdt/reinforcement-learning/tree/master/a2c
        """        
        
        for i in range(self.mem_steps):
            if self.done:
                self.reset()

            actions = self.actor.sample_action(self.state)
            clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            
            next_state, reward, self.done, info = self.env.step(clipped_actions)
            self.memory.remember(self.state, actions, reward, next_state, self.done)
            self.state = next_state
            self.steps+=1
            self.episode_reward+=reward

            if self.done:
                self.episode_reward_store.append(self.episode_reward)
                self.writer.write(len(self.episode_reward_store),self.episode_reward, np.mean(self.episode_reward_store[-10:]))
                if len(self.episode_reward_store)%10==0:
                    print('Episode: ', len(self.episode_reward_store), 'Average Episode Reward: ', np.mean(self.episode_reward_store[-10:]))
                       
        return 
        
    def policy_update(self):
        """performs the policy update based on stored experience
        idea from https://github.com/hermesdt/reinforcement-learning/tree/master/a2c
        """        
        
        states, actions, rewards, next_states, dones = self.memory.process_memory(self.gamma)
        
        #convert to pytorch tensors
        states = torch.tensor(np.array(states),dtype=torch.float32).to(self.actor.device)
        next_states = torch.tensor(np.array(next_states),dtype=torch.float32).to(self.actor.device)
        rewards = torch.tensor(rewards,dtype=torch.float32).to(self.actor.device).view(-1,1)
        actions = torch.tensor(actions,dtype=torch.float32).to(self.actor.device)
        dones = torch.tensor(dones,dtype=torch.float32).to(self.actor.device).view(-1,1)
        
        
        if self.algo == 'REINFORCE':
            td_target = rewards #REINFORCE expects td_target = rewards
        else:
            td_target = rewards + self.gamma * self.critic(next_states) * (1-dones) #A2C td_target = rewards + gamma*V(s')
        value = self.critic(states)
        advantage = td_target - value #advantage = td_target - V(s)
        
        #actor loss
        policies = self.actor(states)
        log_probs = policies.log_prob(actions)
        # entropy = policies.entropy().mean() # can try adding entropy to the loss
        
        actor_loss = -(log_probs * advantage).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        
        self.actor.gradient_norm_clip(self.max_grad_norm)
        
        self.actor.optimizer.step()
        
        #critic loss
        critic_loss  = F.mse_loss(td_target, value)
        self.critic.optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        
        self.critic.gradient_norm_clip(self.max_grad_norm)
        
        self.critic.optimizer.step()
        
    def learn(self,total_steps):
        """learns the policy for a fixed number of steps

        Args:
            total_steps (int): number of policy updates to perform

        Returns:
            list: reward history
        """        
        for i in range(total_steps):
            self.policy_evalulation()
            self.policy_update()
            # if i%1000==0:
            #     self.save_models()
            if np.mean(self.episode_reward_store[-10:])> self.best_score:
                self.best_score = np.mean(self.episode_reward_store[-10:])
                self.save_models()
        self.env.close()
        return self.episode_reward_store
    
    def evaluate(self,total_steps=5000):
        """evaluates the policy for a fixed number of episodes

        Args:
            render (bool, optional): whether to render the environment. Defaults to False.

        Returns:
            list: reward history
        
        """        
             
        for i in range(total_steps):
            if self.done:
                self.reset()

            actions = self.actor.sample_action(self.state)
            clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            
            next_state, reward, self.done, info = self.env.step(clipped_actions)
            self.memory.remember(self.state, actions, reward, next_state, self.done)
            self.state = next_state
            self.steps+=1
            self.episode_reward+=reward

            if self.done:
                self.episode_reward_store.append(self.episode_reward)
                self.writer.write(len(self.episode_reward_store),self.episode_reward, np.mean(self.episode_reward_store[-10:]))
                if len(self.episode_reward_store)%10==0:
                    print('Episode: ', len(self.episode_reward_store), 'Average Episode Reward: ', np.mean(self.episode_reward_store[-10:]))
        
        return self.episode_reward_store

    def save_models(self):
        """save the actor and critic networks
        """        
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        
    def load_models(self):
        """load the actor and critic networks
        """        
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        
        
    
        
        
        
        
        

