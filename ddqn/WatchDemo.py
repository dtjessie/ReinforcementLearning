import argparse
from time import sleep
import gym
import imageio
import numpy as np
from DQAgent import DQAgent


def main(environment, file_out, weight_file, action_value, f_duration, watch, save):
    use_CNN = False
    env = gym.make(environment)
    if use_CNN is True:
        state_size = (88, 80, 1)
    else:
        state_size = env.observation_space.shape[0]

    action_size = env.action_space.n

    online_dqn = DQAgent(state_size, action_size, loss="huber_loss", action=action_value, use_CNN=use_CNN)
    target_dqn = DQAgent(state_size, action_size, loss="huber_loss", action=action_value, use_CNN=use_CNN)
    online_dqn.model.load_weights(weight_file)
    target_dqn.update_target_weights(online_dqn.model)

    print("Playing {} using weights {} and action {}").format(environment, weight_file, action_value)

    epsilon_max = .1
    online_dqn.epsilon = epsilon_max
    done = False

    state = env.reset()
    state = np.reshape(state, [1, state_size])
    cumulative_reward = 0
    global_step = 0

    if save is True:
        images = []
    while not done:
        global_step += 1
        # Use target_dqn to make Q-values
        # online_dqn then takes epsilon-greedy action
        q_values = target_dqn.model.predict(state)[0]

        action = online_dqn.action(q_values, online_dqn.epsilon)

        next_state, reward, done, info = env.step(action)

        next_state = np.reshape(next_state, [1, state_size])
        cumulative_reward += reward

        if watch is True:
            env.render()
        if save is True:
            images.append(env.render(mode="rgb_array"))

        state = next_state

        if done:
            print("Score {}, Total steps {}").format(cumulative_reward, global_step)
            break
    if save is True:
        imageio.mimsave(file_out, images, duration=f_duration)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0")
    parser.add_argument("--out", type=str, default="Testing.gif")
    parser.add_argument("--weight_file", type=str, default="CartPole-v0_online_dqn_solved.h5")
    parser.add_argument("--action_value", type=str, default="epsilon_greedy")
    parser.add_argument("--duration", type=float, default=.06)
    parser.add_argument("--watch", type=bool, default=True)
    parser.add_argument("--save", type=bool, default=True)
    args = parser.parse_args()
    main(args.env, args.out, args.weight_file, args.action_value, args.duration, args.watch, args.save)
