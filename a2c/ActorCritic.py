import keras
import numpy as np
from LossFunctions import *

class Actor:
    def __init__(self, state_size, action_size, reward_discount, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        #self.i = 1.0 # From Sutton and Barto, but other not used other places
        self.gamma = reward_discount        
        self.learning_rate = learning_rate
        
        self.loss = "categorical_crossentropy"
        self.model = self.build_DQN()
        
    def build_DQN(self):
        # Simple neural network. Cart pole has only a 4-d input space
        input_state = keras.Input(shape = (self.state_size, ))
        x = keras.layers.Dense(24, activation = 'relu')(input_state)
        x = keras.layers.Dense(24, activation = 'relu')(x)
        output_action_probs = keras.layers.Dense(self.action_size, activation = 'softmax')(x)
        model = keras.models.Model(input_state, output_action_probs)
        model.compile(optimizer = keras.optimizers.Adam(lr = self.learning_rate),
                      loss = self.loss)
        return model
        
        ###########################################
        # Below is usual CNN architecture
        ###########################################
        #input_state = keras.Input(shape = (88, 80, 1, )) # (self.state_size, ))
        #x = keras.layers.Conv2D(32, (8,8), strides = (4,4), padding ="same")(input_state)
        #x = keras.layers.Conv2D(64, (4,4), strides = (2,2), padding ="same")(x)
        #x = keras.layers.Conv2D(128, (3,3), strides = (2,2), padding ="same")(x)
        #x = keras.layers.Flatten()(x)
        #x = keras.layers.Dense(512, activation = "relu")(x)
        #output_action_probs = keras.layers.Dense(self.action_size, activation = "softmax")(x)
        #model = keras.models.Model(input_state, output_action_probs)
        #model.compile(optimizer = keras.optimizers.Adam(lr = self.learning_rate),
        #              loss = self.loss)
        #return model
    
    def act(self, state):
        act_probs = self.model.predict(state)
        return [np.random.choice(self.action_size, 1, p = probs)[0] for probs in act_probs]
    
    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)
        
class Critic:
    def __init__(self, state_size, value_size, reward_discount, learning_rate):
        self.state_size = state_size
        self.value_size = value_size
        self.gamma = reward_discount        
        self.learning_rate = learning_rate
        
        self.loss = "mse"
        self.model = self.build_DQN()
        
    def build_DQN(self):
        # Simple neural network. Cart pole has only a 4-d input space
        input_state = keras.Input(shape = (self.state_size, ))
        x = keras.layers.Dense(24, activation = 'relu')(input_state)
        x = keras.layers.Dense(24, activation = 'relu')(x)
        output_action_weights = keras.layers.Dense(self.value_size, activation = 'linear')(x)
        model = keras.models.Model(input_state, output_action_weights)
        model.compile(optimizer = keras.optimizers.Adam(lr = self.learning_rate),
                      loss = self.loss)
        return model
        ###########################################
        # Below is usual CNN architecture
        ###########################################
        #input_state = keras.Input(shape = (88, 80, 1, )) # (self.state_size, ))
        #x = keras.layers.Conv2D(32, (8,8), strides = (4,4), padding ="same")(input_state)
        #x = keras.layers.Conv2D(64, (4,4), strides = (2,2), padding ="same")(x)
        #x = keras.layers.Conv2D(128, (3,3), strides = (2,2), padding ="same")(x)
        #x = keras.layers.Flatten()(x)
        #x = keras.layers.Dense(512, activation = "relu")(x)
        #output_action_values = keras.layers.Dense(self.action_size, activation = "linear")(x)
        #model = keras.models.Model(input_state, output_action_values)
        #model.compile(optimizer = keras.optimizers.Adam(lr = self.learning_rate),
        #              loss = self.loss)
        #return model
 
    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)