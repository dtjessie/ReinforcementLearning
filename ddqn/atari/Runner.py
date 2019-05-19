import numpy as np
import gym
from Preprocess import preprocess


class Runner:
    def __init__(self, environment, batch_size, group_size, render=False, make_gif=False):
        self.env = [gym.make(environment) for i in range(batch_size)]
        self.state_size = self.env[0].observation_space.shape
        self.action_size = self.env[0].action_space.n
        self.batch_size = batch_size
        self.group_size = group_size

    def get_state_size(self):
        return self.state_size

    def get_action_size(self):
        return self.action_size

    def reset_all(self):
        """ Returns a stack of 4 copies of the original reset state for
        each runner: return shape is (batch_size, 88, 80, group_size)"""
        reset_env = [env.reset() for env in self.env]
        # (64, 210, 163, 3)
        reset_env = np.array([preprocess(reset_env[i]) for i in range(self.batch_size)])
        # (64, 88, 80, 1)
        reset_env_stack = [reset_env for k in range(self.group_size)]
        # (4, 64, 88, 80, 1)
        reset_env = np.concatenate(reset_env_stack, axis=-1)
        # (64, 88, 80, 4)
        return reset_env

    def reset_one(self, i):
        reset_state = self.env[i].reset()
        reset_state = preprocess(reset_state)
        reset_state = np.array(np.concatenate([reset_state for k in range(self.group_size)], axis=-1))
        return reset_state

    def step(self, action):
        next_state = [[] for empty in range(self.group_size)]
        reward_sum = np.zeros(self.batch_size)

        for i in range(self.group_size):
            outcomes = [env.step(act) for env, act in zip(self.env, action)]

            cols = [[], [], [], []]  # next_state, reward, done, info

            for j in range(self.batch_size):
                one_step = outcomes[j]
                for col, value in zip(cols, one_step):
                    col.append(value)
            cols = [np.array(col) for col in cols]

            cols[0] = np.array([preprocess(cols[0][k][:][:]) for k in range(self.batch_size)])
            next_state[i].append(cols[0])
            reward_sum += cols[1]
        # Now next_state has shape (group_size, 1, batch_size, 88, 80, 1)
        # So reshape to (group_size, batch_size, 88, 80, 1)
        # Split them, stack them to get (batch_size, 88, 80, group_size)
        next_state = np.reshape(next_state, [self.group_size, self.batch_size, 88, 80, 1])
        split_states = [next_state[k] for k in range(self.group_size)]
        next_state = np.array(np.concatenate(split_states, axis=-1))

        return next_state, reward_sum, cols[2], cols[3]

    def render(self, mode=None):
        if mode is None:
            self.env[0].render()
        else:
            return self.env[0].render(mode)
