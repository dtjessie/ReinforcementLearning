###########################################################################
# These are different loss functions that can be used for RL training.
# The functions are part of the DQAgent class definition, DQAgent.loss()
# Used to update weights in the DQAgent neural nets that are used to
# compute the Q-values of actions.

import keras

class LossFunctions:
    def __init__(self, loss = "huber_loss"):
        None
            
    @classmethod
    def get_loss_function(cls, loss):
        if loss == "huber_loss":
            return LossFunctions.huber_loss
        elif loss == "geron_loss":
            return LossFunctions.geron_loss
        elif loss == "mse_loss":
            return LossFunctions.mse_loss
        else:
            print("#"*20)
            print("Warning! Unknown loss function: {}").format(loss)
            print("Using geron_loss then")
            print("#"*20)
            return LossFunctions.huber_loss
        
    @classmethod
    def geron_loss(cls, q, y_pred):
        # This function comes from Geron's "Hands-On ML
        # Small errors (<1) are reduced squaring,
        # Large errors (>1) are linear
        error = keras.backend.abs(q - y_pred)
        clip_error = keras.backend.clip(error, 0.0, 1.0)
        linear_error = 2*(error - clip_error)
        return keras.backend.mean(keras.backend.square(error) + linear_error)
    
    @classmethod
    def mse_loss(cls, q, y_pred):
        # Standard mean squared error
        squared_error = keras.backend.square(q - y_pred)
        return keras.backend.mean(squared_error)

    @classmethod
    def huber_loss(cls, q, y_pred):
        # This loss function became popular after the DQN nature
        # paper from DeepMind. 
        # Small errors (<1) are reduced by squaring,
        # Large errors (>1) are linear
        # Then glued together to keep it continuous
        error = keras.backend.abs(q - y_pred)
        #clip_error = keras.backend.clip(error, 0.0, 1.0)
        linear_error = error - .5
        use_linear_flag = keras.backend.cast((error > 1), 'float32')
        return keras.backend.mean((use_linear_flag * linear_error + .5*(1-use_linear_flag) * keras.backend.square(error)))