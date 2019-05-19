import argparse
import gym
import numpy as np
import os
from collections import deque
from datetime import datetime
from DQAgent import DQAgent
from Runner import Runner

FIRE_ACTION_NUMBER = 1


def sample_memories(replay_memory, batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []]  # state, action, reward, next_state, done
    for i in indices:
        memory = replay_memory[i]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]

    return (cols[0].reshape(batch_size, 88, 80, 4), cols[1], cols[2], cols[3].reshape(batch_size, 88, 80, 4), cols[4])


def replay_from_memory(online_dqn, target_dqn, batch_size, num_epochs=3):
    # Sample a batch_size number of memories
    # Use the target_dqn to predict q-values
    # Use the online_dqn to fit to predicted q-values
    # Run for NUM_EPOCHS!

    if len(online_dqn.memory) < batch_size:
        batch_size = len(online_dqn.memory)

    state, action, reward, next_state, done = sample_memories(online_dqn.memory, batch_size)

    ##########################################################################################
    # The below version uses one interpretation of the Double Q-Learning from Sutton and Barto
    # There is another (better?) interpretation below, and is the one used in DeepMind's Nature
    # paper. Difference is in how to compute next_state_best_action. The method below uses
    # the online network to compute the next_state's best action, which is suspect.
    ##########################################################################################

    target = target_dqn.model.predict(state)

    next_state_best_action = np.argmax(online_dqn.model.predict(next_state), axis=1)
    q_value_next_state = target_dqn.model.predict(next_state)

    for idx, act, next_act, done_flag in zip(range(batch_size), action, next_state_best_action, done):

        if done_flag:
            target[idx][act] = reward[idx]
        else:
            target[idx][act] = reward[idx] + online_dqn.gamma * q_value_next_state[idx][next_act]
    online_dqn.model.fit(state, target, epochs=num_epochs, verbose=0)

    ##########################################################################################
    # This is the version used in DeepMind's Nature paper. This uses the target network to
    # compute the next_state's best action. Which way works better?
    ##########################################################################################
    #target = target_dqn.model.predict(state)

    #q_value_next_state = target_dqn.model.predict(next_state)

    #for idx, act, done_flag in zip(range(batch_size), action, done):

        #if done_flag:
            #target[idx][act] = reward[idx]
        #else:
            #target[idx][act] = reward[idx] + online_dqn.gamma * np.amax(q_value_next_state[idx])
    #online_dqn.model.fit(state, target, epochs=num_epochs, verbose=0)


def setup_logs(environment, time_stamp, base_dir='./logs/'):
    # Make log directories, maybe even ./logs/... directories
    log_dir = os.path.join(base_dir, environment, time_stamp)

    try:
        os.mkdir(log_dir)
    except OSError as error:
        print(error)
        print("Making log directories {}").format(os.path.join(base_dir, environment))
        try:
            os.mkdir(os.path.join(base_dir, environment))
            os.mkdir(log_dir)
        except OSError as error:
            print error
            print("Making directories {}").format(os.path.join(base_dir, environment, log_dir))
            os.mkdir(base_dir)
            os.mkdir(os.path.join(base_dir, environment))
            os.mkdir(log_dir)

    try:
        parameter_file = os.open(os.path.join(log_dir, "parameters"), os.O_RDWR | os.O_CREAT)
        score_file = os.open(os.path.join(log_dir, "score_history"), os.O_RDWR | os.O_CREAT)
    except OSError as error:
        print(error)

    return log_dir, parameter_file, score_file


def main(environment, loss_function, action_value, use_CNN,
         total_games, max_time_per_game, burn_in,
         training_interval, target_update_interval, save_interval,
         num_epochs, batch_size, learning_rate,
         epsilon_max, epsilon_min, epsilon_decay_steps, gamma, memory_size, log_interval):
    # Set up logging
    start_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log_dir, parameter_file, score_file = setup_logs(environment, start_time)

    ################################
    # Save our training parameters #
    line = "loss_function: {}\nactionvalue: {}\ntotal_games: {}\nmax_time_per_game: {}\ntraining_interval: {}\ntarget_update_interval: {}\nsave_interval: {}\nnum_epochs: {}\nbatch_size: {}\nlearning_rate: {}\nepsilon_max: {}\nepsilon_min: {}\nepsilon_decay_steps: {}\ngamma: {}\nmemory_size: {}\nlog_interval: {}\n".format(loss_function, action_value, total_games, max_time_per_game, training_interval, target_update_interval,
            save_interval, num_epochs, batch_size, learning_rate, epsilon_max, epsilon_min, epsilon_decay_steps, gamma, memory_size, log_interval)
    os.write(parameter_file, line)
    ################################

    # Set up our environment
    env = gym.make(environment)
    if use_CNN is True:
        state_size = (88, 80, 1)
    else:
        state_size = env.observation_space.shape[0]

    action_size = env.action_space.n

    # Stack group_size number of atari images
    group_size = 4

    # The following are hard-coded for now, but original image
    # is scaled by preprocssing down to 88, 80, 1 and we combine
    # 4 of them to get a batch of images
    # Note that the "1" argument is the number of copies of environment to train simultaneously
    runner = Runner(environment, 1, group_size)

    # Note that if use_CNN = True, then the state_size is ignored!
    online_dqn = DQAgent(state_size, action_size, loss=loss_function, action=action_value, learning_rate=learning_rate,
                         epsilon=epsilon_max, gamma=gamma, memory_size=memory_size, use_CNN=use_CNN)
    target_dqn = DQAgent(state_size, action_size, loss=loss_function, action=action_value, learning_rate=learning_rate,
                         epsilon=epsilon_max, gamma=gamma, memory_size=memory_size, use_CNN=use_CNN)

    target_dqn.update_target_weights(online_dqn.model)

    # Solved criterion for CartPole, LunarLander, etc
    if environment == "CartPole-v0":
        solved_thresh = 195.0
        max_time_per_game = 200
    elif environment == "LunarLander-v2":
        solved_thresh = 200.0
        max_time_per_game = 1000
    else:
        solved_thresh = 500.0
        print("Not sure solution condition for {}; using average of 100 rounds > {}".format(environment, solved_thresh))

    print("Playing {} using loss {} and action {}").format(environment, loss_function, action_value)

    done = False
    score_history = deque([], maxlen=log_interval)
    max_score = 0
    global_step = 0
    game_num = 1

    state = runner.reset_all()
    cumulative_reward = 0
    lives = 5
    done_flags = True

    while game_num < total_games:
        # Use target_dqn to make Q-values
        # online_dqn then takes epsilon-greedy action
        global_step += 1

        q_values = online_dqn.model.predict(state)[0]

        # If we lose a life, start with a few FIRE actions
        # to get started again. Random to avoid learning
        # fixed sequence of actions
        if done_flags is False:
            action = online_dqn.action(q_values, online_dqn.epsilon)
        else:
            random_fire_actions = np.random.randint(1, 3)
            for i in range(random_fire_actions):
                action = FIRE_ACTION_NUMBER
                next_state, reward, done, info = runner.step([action])
            state = next_state
            done_flags = False
            continue

        next_state, reward, done, info = runner.step([action])
        cumulative_reward += reward[0]

        # Losing a life is bad, so say so
        remaining_lives = info[0]["ale.lives"]
        life_lost_flag = bool(lives - remaining_lives)
        lives = remaining_lives

        done_flags = False
        if life_lost_flag or done:
            done_flags = True

        # Store the result in memory so we can replay later
        online_dqn.remember(state, action, reward, next_state, done_flags)
        state = next_state

        if done:
            score_history.append(cumulative_reward)

            if cumulative_reward > max_score:
                max_score = cumulative_reward

            if game_num % log_interval == 0:
                os.write(score_file, str(list(score_history))+'\n')
                print("Completed game {}/{}, global step {}, last {} games average: {:.3f}, max: {}, min: {}. Best so far {}. Epsilon: {:.3f}".format(game_num, total_games, global_step, log_interval, np.average(score_history), np.max(score_history), np.min(score_history), max_score, online_dqn.epsilon))

            game_num += 1
            cumulative_reward = 0
            lives = 5
            state = runner.reset_all()

            # If we have an average score > 195.0 over 100 consecutive rounds, we have solved CartPole!
            if game_num > 100:
                avg_last_100 = np.average(score_history)

                if avg_last_100 > solved_thresh:
                    stop_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
                    print("Congratulations! {} has been solved after {} games.").format(environment, game_num)
                    online_dqn.model.save(os.path.join(log_dir, "online_dqn_{}_solved.h5".format(environment)))
                    line = "Training start: {}\nTraining ends:  {}\n".format(start_time, stop_time)
                    os.write(parameter_file, line)
                    os.write(score_file, str(list(score_history))+'\n')
                    os.close(parameter_file)
                    os.close(score_file)
                    return 0

        # For the first burn_in number of rounds, just populate memory
        if global_step < burn_in:
            continue
        # Once we are past the burn_in exploration period, we start to train
        # This is a linear decay that goes from epsilon_max to epsion_min in epsilon_decay_steps
        online_dqn.epsilon = max(epsilon_max + ((global_step-burn_in)/float(epsilon_decay_steps))*(epsilon_min-epsilon_max), epsilon_min)

        if (global_step % training_interval == 0):
            replay_from_memory(online_dqn, target_dqn, batch_size, num_epochs)

        if (global_step % target_update_interval == 0):
            target_dqn.update_target_weights(online_dqn.model)

        if global_step % save_interval == 0:
            online_dqn.model.save(os.path.join(log_dir, "online_dqn" + ".h5"))

    ##################################################################
    # If we're here, then we finished our training without solution #
    # Let's save the most recent models and make the plots anyway   #
    #################################################################
    stop_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    online_dqn.model.save(os.path.join(log_dir, "online_dqn_" + str(global_step) + ".h5"))

    print("Done! Completed game {}/{}, global_step {}".format(game_num, total_games, global_step))
    line = "\n \nTraining start: {}\nTraining ends:  {}\n \n".format(start_time, stop_time)
    os.write(parameter_file, line)
    if game_num % log_interval != 0:
        os.write(score_file, str(list(score_history)[:game_num % log_interval])+'\n')
    os.close(parameter_file)
    os.close(score_file)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="BreakoutDeterministic-v4")
    parser.add_argument("--loss_function", type=str, default="huber_loss")
    parser.add_argument("--action_value", type=str, default="epsilon_greedy")
    parser.add_argument("--use_CNN", type=bool, default=True)

    parser.add_argument("--total_games", type=int, default=100000)
    parser.add_argument("--max_time_per_game", type=int, default=200)
    parser.add_argument("--burn_in", type=int, default=100000)

    parser.add_argument("--training_interval", type=int, default=4)
    parser.add_argument("--target_update_interval", type=int, default=10000)
    parser.add_argument("--save_interval", type=int, default=10000)

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=.0001)

    parser.add_argument("--epsilon_max", type=float, default=1.0)
    parser.add_argument("--epsilon_min", type=float, default=.1)
    parser.add_argument("--epsilon_decay_steps", type=float, default=1000000)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--memory_size", type=int, default=750000)
    parser.add_argument("--log_interval", type=int, default=100)
    args = parser.parse_args()
    main(args.env, args.loss_function, args.action_value, args.use_CNN,
         args.total_games, args.max_time_per_game, args.burn_in,
         args.training_interval, args.target_update_interval, args.save_interval,
         args.num_epochs, args.batch_size, args.learning_rate,
         args.epsilon_max, args.epsilon_min, args.epsilon_decay_steps, args.gamma, args.memory_size,
         args.log_interval)
