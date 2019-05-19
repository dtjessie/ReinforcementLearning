import argparse
from time import sleep
import gym
import imageio
import numpy as np
from DQAgent import DQAgent
from Runner import Runner


def main(environment, file_out, weight_file, action_value, f_duration, watch, save):
    use_CNN = True
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

    online_dqn = DQAgent(state_size, action_size, loss="huber_loss", action=action_value, use_CNN=True)
    target_dqn = DQAgent(state_size, action_size, loss="huber_loss", action=action_value, use_CNN=True)
    online_dqn.model.load_weights(weight_file)
    target_dqn.update_target_weights(online_dqn.model)

    print("Playing {} using weights {} and action {}").format(environment, weight_file, action_value)

    epsilon_max = .1
    online_dqn.epsilon = epsilon_max
    done = False

    done_flags = True
    lives = 5

    state = runner.reset_all()
    cumulative_reward = 0
    global_step = 0
    if save is True:
        images = []
    while not done:
        global_step += 1
        # Use target_dqn to make Q-values
        # online_dqn then takes epsilon-greedy action
        q_values = target_dqn.model.predict(state)[0]

        if done_flags is False:
            action = online_dqn.action(q_values, online_dqn.epsilon)
        else:
            random_fire_actions = np.random.randint(1, 3)
            for i in range(random_fire_actions):
                action = 1
                next_state, reward, done, info = runner.step([action])
            state = next_state
            done_flags = False
            continue

        next_state, reward, done, info = runner.step([action])
        if watch is True:
            runner.render()
            sleep(.05)
        if save is True:
            images.append(runner.render(mode="rgb_array"))
        cumulative_reward += reward

        # Losing a life is bad, so say so
        remaining_lives = info[0]["ale.lives"]
        life_lost_flag = bool(lives - remaining_lives)
        lives = remaining_lives

        done_flags = False
        if life_lost_flag or done:
            done_flags = True

        state = next_state

        if done:
            print("Score {}, Total steps {}").format(cumulative_reward, global_step)
            break
    if save is True:
        imageio.mimsave(file_out, images, duration=f_duration)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="BreakoutDeterministic-v4")
    parser.add_argument("--out", type=str, default="Testing.gif")
    parser.add_argument("--weight_file", type=str, default="Breakout_online_dqn.h5")
    parser.add_argument("--action_value", type=str, default="epsilon_greedy")
    parser.add_argument("--duration", type=float, default=.06)
    parser.add_argument("--watch", type=bool, default=True)
    parser.add_argument("--save", type=bool, default=True)
    args = parser.parse_args()
    main(args.env, args.out, args.weight_file, args.action_value, args.duration, args.watch, args.save)
