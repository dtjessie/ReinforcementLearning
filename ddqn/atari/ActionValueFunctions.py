###########################################################################
# These are different action-value functions that can be used for RL 
# training. The functions are part of the DQAgent class definition, 
# DQAgent.action()
# Given Q-values, how to choose the next action?

import keras
import numpy as np

class ActionValueFunctions:
    def __init__(self, action = "epsilon_greedy"):
        None

    @classmethod
    def get_action_function(cls, action):
        if action == "epsilon_greedy":
            return ActionValueFunctions.epsilon_greedy
        elif action == "gradient_choice":
            print("#"*20)
            print("Warning! This seems to work fine for a little bit, then gives an overflow error.")
            print("#"*20)
            return ActionValueFunctions.gradient_choice
        else:
            print("Unknown action-value function: {}").format(action)
            print("Using epsilon-greedy then")
            return ActionValueFunctions.epsilon_greedy
    
    @classmethod
    def epsilon_greedy(cls, q_values, epsilon):
        # Greedy: choose the action with the highest Q value
        # Epsilon: epsilon chance of choosing randomly
        if (np.random.rand() < epsilon):
            return np.random.randint(0, len(q_values))
        return np.argmax(q_values)
    
    @classmethod
    def gradient_choice(cls, q_values):
        # Choose randomly based on relative weights of Q values
        weights = np.exp(q_values)
        weight_total = np.sum(weights)
        choice_probs = weights / (weight_total +.0000000001)
        action = np.random.multinomial(1, choice_probs[:-1], 1)
        return np.argmax(action)