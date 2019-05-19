# What is Q-Learning?

Recall that an agent learning what to do in an environment needs to solve the explore/exploit problem and to find a way to assign rewards to the actions in any state. Q-learning is one approach to this and DDQN is a modern deep learning version. The name Q-learning comes from the mathematical notation: Q(S,A) is the value of action A in state S.

The fundamental part of Q-learning is the credit assignment task: how to value an action in a given state, i.e., what is Q(S,A)? If we are in state S, take action A, and get reward R, the value of the action should certainly include R. However, the action A also changes the state to S' and the next reward in this state, R'. Therefore, the reward for action A in S should also include the future reward from S'. Roughly we have the value of an action is the sum of all future rewards. However, since the effect of action A becomes less influential as the states progress (decision on round 1 has more of an effect on the rewards in round 2 than it does in round 1000), the rewards are discounted by a factor gamma < 1. 

The explore/exploit routine has a number of solutions but the most common one is the simple epsilon-greedy approach. This method says to choose the action greedily most of the time: whichever action has the highest Q-value, choose it. However, for an epsilon proportion of the time, we should explore and so we choose an action randomly indepdenent of the Q-value (choose from a uniform distribution). A method for choosing actions based on their value is called an action value function. 

The modern approach of DDQN uses deep neural networks to estimate the Q-values and generate actions. Additional improvements to this basic idea are memory replay and using two networks: one to estimate the Q-values (target network) and one to learn from the environment (online network).

## How to use

The main file is the DDQN_Learning.py file that implements the above routines. It can be run from the command line specifying parameters of the training including different loss functions, etc:

``` 
python DDQN_Learning.py --help

usage: DDQN_Learning.py [-h] [--env ENV] [--loss_function LOSS_FUNCTION]
                        [--action_value ACTION_VALUE]
                        [--total_games TOTAL_GAMES]
                        [--max_time_per_game MAX_TIME_PER_GAME]
                        [--training_interval TRAINING_INTERVAL]
                        [--target_update_interval TARGET_UPDATE_INTERVAL]
                        [--save_interval SAVE_INTERVAL]
                        [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
                        [--learning_rate LEARNING_RATE]
                        [--epsilon_max EPSILON_MAX]
                        [--epsilon_min EPSILON_MIN]
                        [--epsilon_decay EPSILON_DECAY] [--gamma GAMMA]
                        [--memory_size MEMORY_SIZE]
```
## CartPole-v0

This is the simple first environment to test algorithms in. Even with an algorithm that solves CartPole (solving means an average score of at least 195 in 100 consecutive rounds), there can be issues. For example, with the DDQN code above we run with parameters
~~~
loss_function: geron_loss
actionvalue: epsilon_greedy
total_games: 500
max_time_per_game: 200
training_interval: 4
target_update_interval: 40
save_interval: 1000
num_epochs: 3
batch_size: 128
learning_rate: 0.001
epsilon_max: 1.0
epsilon_min: 0.01
epsilon_decay: 0.999
gamma: 0.99
memory_size: 100000

~~~
and we get the results:

![alt text](https://github.com/dtjessie/RL_Experiments/blob/master/ddqn/logs/CartPole-v0/Solved.png "CartPole winning")

Looks encouraging! CartPole was solved, but let's run it again with the same parameters:


![alt text](https://github.com/dtjessie/RL_Experiments/blob/master/ddqn/logs/CartPole-v0/Unsolved.png "CartPole losing")

Complete failure to learn anything!

This type of phenomenon is common in deep RL and is one of the reasons this area can be so frustrating. 
