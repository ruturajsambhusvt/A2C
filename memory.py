import numpy as np


class Memory(object):
    def __init__(self) -> None:
        self.state = []
        self.action = []
        self.new_state = []
        self.reward = []
        self.done = []

    def remember(self, state, action, reward, new_state, done):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.new_state.append(new_state)
        self.done.append(done)

    def recall(self):
        return np.array(self.state), np.array(self.action), np.array(self.reward), np.array(self.new_state), np.array(self.done)

    def clear_memory(self):
        self.state = []
        self.action = []
        self.reward = []
        self.new_state = []
        self.done = []
        
    def discounted_rewards(self,rewards,dones,gamma):
        discounted_rewards = []
        running_ret = 0
        for reward, done in zip(rewards[::-1],dones[::-1]):
            if done:
                running_ret = 0
            running_ret = running_ret*gamma + reward
            discounted_rewards.append(running_ret)
        return discounted_rewards[::-1]
    
    def process_memory(self,gamma=0.99, algo='REINFORCE'):
        states, actions, rewards, new_states, dones = self.recall()
        if algo == 'REINFORCE':
            rewards = self.discounted_rewards(rewards,dones,gamma)
        self.clear_memory()
        return states, actions, rewards, new_states, dones
        
        
    
