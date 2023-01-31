import time

import gym
import numpy as np

import utils.prints as print_utils

N_ACTIONS = 4
N_STATES = 16

LEARNING_RATE = .5
DISCOUNT_RATE = .98

N_EPISODES = 2000

def main():
    """Main"""
    frozone_lake_env = gym.make("FrozenLake-v0")

    Q = np.zeros([N_STATES, N_ACTIONS])
    rewards = []

    for i in range(N_EPISODES):
        state = frozone_lake_env.reset()
        episode_reward = 0
        done = False

        while not done:
            noise = np.random.randn(1, N_ACTIONS) / (i + 1)
            action = np.argmax(Q[state, :] + noise)

            new_state, reward, done, _ = frozone_lake_env.step(action)

            reward = -1 if done and reward < 1 else reward

            Q[state, action] = (
                1 - LEARNING_RATE) * Q[state, action] + LEARNING_RATE * (
                    reward + DISCOUNT_RATE * np.max(Q[new_state, :]))

            episode_reward += reward
            state = new_state

        rewards.append(episode_reward)

    print("Score over time: " + str(sum(rewards) / N_EPISODES))
    print("Final Q-Table Values")

    for i in range(10):
        state = frozone_lake_env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = np.argmax(Q[state, :])
            new_state, reward, done, _ = frozone_lake_env.step(action)
            print_utils.clear_screen()
            frozone_lake_env.render()
            time.sleep(.1)

            episode_reward += reward
            state = new_state

            if done:
                print("Episode Reward: {}".format(episode_reward))
                print_utils.print_result(episode_reward)

        rewards.append(episode_reward)

    frozone_lake_env.close()