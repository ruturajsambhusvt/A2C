import numpy as np


class Memory(object):
    """class for storing policy evaluation data

    Args:
        object (_type_): _description_
    """    
    def __init__(self,algo) -> None:
        """constructor for Memory class

        Args:
            algo (string): either 'A2C' or 'REINFORCE'
        """        
        self.state = []
        self.action = []
        self.new_state = []
        self.reward = []
        self.done = []
        self.algo = algo

    def remember(self, state, action, reward, new_state, done):
        """stores the data from the policy evaluation as state, action, reward, new_state, done

        Args:
            state (np array): state of the environment
            action (np array): action taken by the agent
            reward (np array): reward obtained by the agent
            new_state (np array): new state of the environment
            done (bool): done flag
        """        
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.new_state.append(new_state)
        self.done.append(done)
        

    def recall(self):
        """recalls the data stored in the memory

        Returns:
            np arrays:  state, action, reward, new_state, done arrays
        """        
        return np.array(self.state), np.array(self.action), np.array(self.reward), np.array(self.new_state), np.array(self.done)

    def clear_memory(self):
        """clears the memory
        """        
        self.state = []
        self.action = []
        self.reward = []
        self.new_state = []
        self.done = []
        
    def discounted_rewards(self,rewards,dones,gamma):
        """calculates the discounted returns for REINFORCE- idea from https://github.com/hermesdt/reinforcement-learning/tree/master/a2c

        Args:
            rewards (np array): array of rewards
            dones (np array ): array of done flags
            gamma (float): discount factor

        Returns:
            np array: array of discounted returns
        """        
        discounted_rewards = []
        running_ret = 0
        for reward, done in zip(rewards[::-1],dones[::-1]):
            if done:
                running_ret = 0
            running_ret = running_ret*gamma + reward
            discounted_rewards.append(running_ret)
        return discounted_rewards[::-1]
    
    def process_memory(self,gamma=0.99):
        """calculates the appropriate discounted returns for A2C and REINFORCE

        Args:
            gamma (float, optional): discount factor . Defaults to 0.99.

        Returns:
            np arrays: state, action, discounted returns, new_state, done arrays
        """        
        states, actions, rewards, new_states, dones = self.recall()
        if self.algo == 'REINFORCE':
            rewards = self.discounted_rewards(rewards,dones,gamma)
        self.clear_memory()
        return states, actions, rewards, new_states, dones
        
        
    
