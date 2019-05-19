import keras
from keras.initializers import VarianceScaling
from LossFunctions import *
from ActionValueFunctions import *
from collections import deque


class DQAgent:
    def __init__(self, state_size, action_size, loss="geron_loss", action="epsilon_greedy", learning_rate=.001,
                 epsilon=1.0, gamma=.99, memory_size=1000000, use_CNN=False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque([], maxlen=memory_size)
        self.gamma = gamma    # discount factor for future rewards
        self.epsilon = epsilon  # will be updated, used in epsilon-greedy algorithm
        self.learning_rate = learning_rate

        self.loss = LossFunctions.get_loss_function(loss)
        self.action = ActionValueFunctions.get_action_function(action)
        self.use_CNN = use_CNN
        self.model = self.build_DQN(self.use_CNN)

    def build_DQN(self, use_CNN):
        # Simple neural network. Cart pole has only a 4-d input space
        if use_CNN is False:
            input_state = keras.Input(shape=(self.state_size, ))
            x = keras.layers.Dense(24, activation='relu')(input_state)
            x = keras.layers.Dense(24, activation='relu')(x)
            output_action_weights = keras.layers.Dense(self.action_size, activation='linear')(x)
            model = keras.models.Model(input_state, output_action_weights)
            model.compile(optimizer=keras.optimizers.Adam(lr=self.learning_rate),
                          loss=self.loss)
            return model
        # This is used for Atari games that are resized.
        # Note the input is hard-coded to (88, 80, 4, )!
        else:
            input_state = keras.Input(shape=(88, 80, 4, ))
            x = keras.layers.BatchNormalization()(input_state)
            x = keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding="valid", use_bias=False, kernel_initializer=VarianceScaling(scale=2.0))(x)
            x = keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding="valid", use_bias=False, kernel_initializer=VarianceScaling(scale=2.0))(x)
            x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding="valid", use_bias=False, kernel_initializer=VarianceScaling(scale=2.0))(x)
            x = keras.layers.Flatten()(x)
            x = keras.layers.Dense(512, activation="relu", use_bias=False, kernel_initializer=VarianceScaling(scale=2.0))(x)
            output_action_weights = keras.layers.Dense(self.action_size, activation="linear")(x)
            model = keras.models.Model(input_state, output_action_weights)
            model.compile(optimizer=keras.optimizers.Adam(lr=self.learning_rate),
                          loss=self.loss)
            return model

    def update_target_weights(self, online_model):
        # Target DQN copies weights from Online DQN
        self.model.set_weights(online_model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
