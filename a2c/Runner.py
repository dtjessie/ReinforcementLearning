import numpy as np
import gym

class Runner:
    def __init__(self, environment, batch_size):
        self.env = [gym.make(environment) for i in range(batch_size)]
        self.state_size = self.env[0].observation_space.shape[0]
        self.action_size = self.env[0].action_space.n
        self.batch_size = batch_size
        
    def get_state_size(self):
        return self.state_size
    
    def get_action_size(self):
        return self.action_size
    
    def reset_all(self):
        return [env.reset() for env in self.env]
    
    def reset_one(self, i):
        return self.env[i].reset()
        
    def step(self, action):
        outcomes = [env.step(act) for env, act in zip(self.env, action)]
        cols = [ [], [], [], [] ] # next_state, reward, done, info
        
       
        for i in range(self.batch_size):
            one_step = outcomes[i]
            for col, value in zip(cols, one_step):
                col.append(value)
        cols = [np.array(col) for col in cols]
        
        #state_size = cols[0].shape[2]
        #return cols[0].reshape(self.batch_size, self.state_size), cols[1], cols[2], cols[3].reshape(self.batch_size, self.state_size), cols[4]
        return cols[0], cols[1], cols[2], cols[3]
    
    def render(self, mode = None):
        return self.env[0].render(mode)