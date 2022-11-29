import torch  
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributions
import numpy as np
from Network import ActorNetwork, CriticNetwork
from memory import Memory

class Agent(object):
    def __init__(self,env,alpha,beta,layer1_critic=256,layer2_critic=256,layer1_actor=256,layer2_actor=256, gamma=0.99,mem_steps=32,algo='REINFORCE',max_grad_norm=0.5) -> None:
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
        
        self.mem_steps = mem_steps
        self.state = None
        self.done = True
        self.steps = 0
        self.episode_reward = 0
        self.episode_reward_store  = []
        self.algo = algo
        self.max_grad_norm = max_grad_norm
    
        
        self.memory = Memory(self.algo)
        self.actor = ActorNetwork(alpha=alpha, input_dim=self.state_dim, action_dim= self.action_dim, max_action= self.env.action_space.high,fc1_size= self.layer1_actor, fc2_size= self.layer2_actor, name = 'Actor'+ self.env_name)
        self.critic = CriticNetwork(beta=beta, input_dim= self.state_dim, fc1_size= self.layer1_critic, fc2_size= self.layer2_critic, name =  'Critic' + self.env_name)
        
    # def choose_action(self,state):
    #     with torch.no_grad():
    #         state = torch.tensor(np.array([state]),dtype=torch.float32).to(self.actor.device)
    #         action , self.log_probs = self.actor.sample_normal_dist(state)
    #         return action.detach().cpu().numpy().flatten()
    def reset(self):
        self.state = self.env.reset()
        self.done = False
        self.episode_reward = 0
        
    def policy_evalulation(self):
        
        for i in range(self.mem_steps):
            if self.done:
                self.reset()
            
            # dists = self.actor(torch.tensor(np.array(self.state),dtype=torch.float32).to(self.actor.device))
            # actions = dists.sample().detach().cpu().numpy().flatten()
            actions = self.actor.sample_action(self.state)
            clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            
            next_state, reward, self.done, info = self.env.step(clipped_actions)
            self.memory.remember(self.state, actions, reward, next_state, self.done)
            self.state = next_state
            self.steps+=1
            self.episode_reward+=reward

            if self.done:
                self.episode_reward_store.append(self.episode_reward)
                if len(self.episode_reward_store)%10==0:
                    print('Episode: ', len(self.episode_reward_store), 'Episode Reward: ', self.episode_reward)
                    
        return 

                
            
        
    def policy_update(self):
        
        states, actions, rewards, next_states, dones = self.memory.process_memory(self.gamma)
        
        states = torch.tensor(np.array(states),dtype=torch.float32).to(self.actor.device)
        next_states = torch.tensor(np.array(next_states),dtype=torch.float32).to(self.actor.device)
        rewards = torch.tensor(rewards,dtype=torch.float32).to(self.actor.device).view(-1,1)
        actions = torch.tensor(actions,dtype=torch.float32).to(self.actor.device)
        dones = torch.tensor(dones,dtype=torch.float32).to(self.actor.device).view(-1,1)
        
        
        if self.algo == 'REINFORCE':
            td_target = rewards
        else:
            td_target = rewards + self.gamma * self.critic(next_states) * (1-dones)
        value = self.critic(states)
        advantage = td_target - value
        
        #actor loss
        policies = self.actor(states)
        log_probs = policies.log_prob(actions)
        entropy = policies.entropy().mean()
        
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
        for i in range(total_steps):
            self.policy_evalulation()
            self.policy_update()
            if i%1000==0:
                self.save_models()
        self.env.close()
        return self.episode_reward_store
        

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        
    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        
        
    # def learn(self, state, action, reward, next_state, done):
        
    #     state = torch.tensor(np.array([state]),dtype=torch.float32).to(self.actor.device)
    #     next_state = torch.tensor(np.array([next_state]),dtype=torch.float32).to(self.actor.device)
    #     reward = torch.tensor(reward,dtype=torch.float32).to(self.actor.device)
    #     action = torch.tensor(action,dtype=torch.float32).to(self.actor.device)
    #     done = torch.tensor(done,dtype=torch.bool).to(self.actor.device)
        
        
    #     self.actor.optimizer.zero_grad()
    #     self.critic.optimizer.zero_grad()
        
    #     next_critic = self.critic.forward(next_state)
    #     critic = self.critic.forward(state)
        
    #     #TD error for the critic
    #     delta = reward + self.gamma*next_critic*(1-int(done)) - critic
    #     critic_loss = delta**2
        
    #     #Loss for the actor network
    #     actor_loss = -self.log_probs*delta
        
    #     (actor_loss + critic_loss).backward()
        
    #     self.actor.optimizer.step()
    #     self.critic.optimizer.step()
        
    
        
        
        
        
        

