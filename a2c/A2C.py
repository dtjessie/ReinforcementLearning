import numpy as np
import tensorflow as tf
import keras
import gym
from collections import deque
from random import sample
from ActorCritic import *
from Runner import *
from MakePlots import *
import argparse
from datetime import datetime
import os
import errno

def train(actor, critic, state, action, reward, next_state, done, batch_size, num_epochs):

    advantage = np.zeros( (batch_size, actor.action_size) )
    target = np.zeros( (batch_size, critic.value_size) )

    value = critic.model.predict(state)
    next_value = critic.model.predict(next_state)

    for idx, act, done_flag, in zip(range(batch_size), action, done):
        if done_flag:
            next_value[idx] = 0

        target[idx] = reward[idx] + critic.gamma*next_value[idx][0]
        advantage[idx][act] = reward[idx] + actor.gamma*next_value[idx][0] - value[idx][0]

    actor_history = actor.model.fit(state, advantage, epochs=num_epochs, verbose=0)
    critic_history = critic.model.fit(state, target, epochs=num_epochs, verbose=0)

    #actor.i = actor.i * actor.gamma # used in Sutton and Barto, but not other places

def setup_logs(environment, time_stamp, base_dir='./logs/'):
    # NOTE: This code assumes ./logs/environment/time_stamp/
    # is an existing directory! If not, it will fail
    log_dir = os.path.join(base_dir, environment, time_stamp)

    try:
        os.mkdir(log_dir)
    except OSError as e:
        print("OSError: ", e)

    try:
        log_file = os.open(os.path.join(log_dir, "log"), os.O_RDWR | os.O_CREAT)
    except OSError as e:
        print("OSError: ", e)

    return log_dir, log_file

def main(environment, total_games, max_time_per_game,
         training_interval, save_interval,
         num_epochs, batch_size, actor_lr, critic_lr, actor_gamma, critic_gamma):
    # For logging purposes, we'll make use of this
    start_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log_dir, log_file = setup_logs(environment, start_time)

    # Then set up environment
    if environment == "CartPole-v0":
        solved_thresh = 195.0
    elif environment == "LunarLander-v2":
        solved_thresh = 200.0

    TOTAL_GAMES = 100*batch_size
    MAX_TIME_PER_GAME = max_time_per_game
    SAVE_INTERVAL = save_interval
    GAME_OVER_PENALTY = -100
    NUM_EPOCHS = num_epochs

    ################################
    # Save our training parameters #
    line = "total_games: {}\nmax_time_per_game: {}\ntraining_interval: {}\nsave_interval: {}\nnum_epochs: {}\nbatch_size: {}\nactor_lr: {}\ncritic_lr: {}\nactor_gamma: {}\ncritic_gamma: {}\n".format(total_games, max_time_per_game,
                                                                                        training_interval, save_interval,
                                                                                        num_epochs, batch_size,
                                                                                        actor_lr, critic_lr, actor_gamma, critic_gamma)
    os.write(log_file, line)
    ################################

    runner = Runner(environment, batch_size)
    state_size = runner.get_state_size()
    action_size = runner.get_action_size()

    value_size = 1 # This is the output size for the critic model
    actor = Actor(state_size, action_size, actor_gamma, actor_lr)
    critic = Critic(state_size, value_size, critic_gamma, critic_lr)

    done = False
    history = []
    history_avg_100 = []
    game_num = 0
    step = 0
    state = runner.reset_all()
    state = np.reshape(state, [batch_size, state_size])
    cumulative_reward = np.zeros(batch_size)
    game_steps = np.zeros(batch_size)

    print("Playing {} with A2C...").format(environment)

    while (game_num < TOTAL_GAMES):
        step += 1
        game_steps += 1


        action = actor.act(state)
        next_state, reward, done, _ = runner.step(action)
        cumulative_reward += reward

        # Penalize failure harshly
        if environment == "CartPole-v0":
            for i in range(batch_size):
                if (done[i]) and (game_steps[i] < 200):
                    reward[i] = GAME_OVER_PENALTY

        if environment == "LunarLander-v2":
            for i in range(batch_size):
                if game_steps[i] == 1000: # Without this, just hovers to avoid crashing
                    reward[i] = GAME_OVER_PENALTY


        train(actor, critic, state, action, reward, next_state, done, batch_size, num_epochs)
        state = next_state

        for i in range(batch_size):
            if done[i]:
                print("Game {}/{} complete, score {}").format(game_num, TOTAL_GAMES, cumulative_reward[i])
                state[i] = runner.reset_one(i)
                history.append(cumulative_reward[i])
                cumulative_reward[i] = 0
                game_steps[i] = 0
                game_num += 1
               # If we have an average score > solved_thresh over 100 consecutive rounds, we have solved CartPole!
                if len(history) > 100:
                    avg_last_100 = np.average(history[-100:])
                    history_avg_100.append(avg_last_100)

                    if avg_last_100 > solved_thresh:
                        stop_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
                        print("Congratulations! {} been solved after {} rounds.").format(environment, step)
                        actor.model.save(os.path.join(log_dir, "actor_solved_" + str(int(avg_last_100)) + ".h5"))
                        critic.model.save(os.path.join(log_dir, "critic_solved_" + str(int(avg_last_100)) + ".h5"))
                        plot_name = os.path.join(log_dir, "Solved.png")
                        plot_history(history, history_avg_100, plot_name)
                        line = "Training start: {}\nTraining ends:  {}\n".format(start_time, stop_time)
                        os.write(log_file, line)
                        os.write(log_file, "Cumulative score history: \n{}\n\nAverage of 100 rounds: \n{}\n.".format(history, avg_last_100))
                        os.close(log_file)
                        return 0

        # If not, just save the model and keep going    
        if step % SAVE_INTERVAL == 0:
            actor.model.save(os.path.join(log_dir, "actor_" + str(step) + ".h5"))
            critic.model.save(os.path.join(log_dir, "critic_" + str(step) + ".h5"))
    ##################################################################
    # If we're here, then we finished our training without solution #
    # Let's save the most recent models and make the plots anyway   #
    #################################################################
    stop_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    actor.model.save(os.path.join(log_dir, "actor_" + str(step) + ".h5"))
    critic.model.save(os.path.join(log_dir, "critic_" + str(step) + ".h5"))
    plot_name = os.path.join(log_dir, "Unsolved.png")
    plot_history(history, history_avg_100, plot_name)
    line = "Training start: {}\nTraining ends:  {}".format(start_time, stop_time)
    os.write(log_file, line)
    os.write(log_file, "Cumulative score history: \n{}\n\nAverage of 100 rounds: \n{}\n.".format(history, avg_last_100))
    os.close(log_file)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0")

    parser.add_argument("--total_games", type=int, default=200)
    parser.add_argument("--max_time_per_game", type=int, default=1000)
    parser.add_argument("--training_interval", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--actor_lr", type=float, default=.001)
    parser.add_argument("--critic_lr", type=float, default=.005)
    parser.add_argument("--actor_gamma", type=float, default=.999)
    parser.add_argument("--critic_gamma", type=float, default=.999)

    args = parser.parse_args()
    main(args.env, args.total_games, args.max_time_per_game,
        args.training_interval, args.save_interval,
        args.num_epochs, args.batch_size,
        args.actor_lr, args.critic_lr, args.actor_gamma, args.critic_gamma)
