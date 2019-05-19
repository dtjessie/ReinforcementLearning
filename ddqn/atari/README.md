# DDQN for Atari

This is an implementation of [van Hasselt, et al.](https://arxiv.org/abs/1509.06461) to play Breakout. Human performance is an average score of about 32 and this implementation with the paramters below will beat this in about one night of training on a 1080 Ti

This paper has been implemented a lot and there are many tutorials online, but a couple details I found important but not mentioned:

1. A large replay memory is important, but the Atari images can take up size. Save them as np.int8 arrays to save lots of space.

2. Following from above, then include a batch_normalization layer to scale the input 

Still to be implemented is the dueling architecture described in [Wang](https://arxiv.org/abs/1511.06581)

## How to use

The main file is the DDQN_Learning.py file that implements the above routines. It can be run from the command line specifying parameters of the training:

``` 

python DDQN_Learning.py --help

usage: DDQN_Learning.py [-h] [--env ENV] [--loss_function LOSS_FUNCTION]
                        [--action_value ACTION_VALUE] [--use_CNN USE_CNN]
                        [--total_games TOTAL_GAMES]
                        [--burn_in BURN_IN]
                        [--training_interval TRAINING_INTERVAL]
                        [--target_update_interval TARGET_UPDATE_INTERVAL]
                        [--save_interval SAVE_INTERVAL]
                        [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
                        [--learning_rate LEARNING_RATE]
                        [--epsilon_max EPSILON_MAX]
                        [--epsilon_min EPSILON_MIN]
                        [--epsilon_decay_steps EPSILON_DECAY_STEPS]
                        [--gamma GAMMA] [--memory_size MEMORY_SIZE]
                        [--log_interval LOG_INTERVAL]
```
## Parameter values

These parameters will give a reasonable outcome with a night of training. Most of these are the default in the DDQ_Learning.py file above.

~~~
loss_function: huber_loss
actionvalue: epsilon_greedy
total_games: 100000
training_interval: 4
target_update_interval: 10000
save_interval: 10000
num_epochs: 1
batch_size: 32
learning_rate: 0.0001
epsilon_max: 1.0
epsilon_min: 0.1
epsilon_decay_steps: 1000000
gamma: 0.99
memory_size: 750000
log_interval: 100
~~~
Typical play after a day or so:
![Typical result](https://github.com/dtjessie/ReinforcementLearning/blob/master/ddqn/atari/Breakout_example.gif)
A bit higher score:
![Higher score](https://github.com/dtjessie/ReinforcementLearning/blob/master/ddqn/atari/Breakout_257.gif)
