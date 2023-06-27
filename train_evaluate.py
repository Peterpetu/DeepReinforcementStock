import torch
import numpy as np

def train(agent, data, num_episodes=1000, target_network_update_freq=5):
    for e in range(num_episodes):
        state = data[0]
        for t in range(1, len(data)):
            action = agent.act(state, training=True)
            next_state = data[t]
            reward = next_state[-2] - state[-2]  # Calculate reward based on the difference between the closing prices
            done = 1 if t == len(data) - 1 else 0
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            agent.replay()

        agent.decay_epsilon()

        if (e + 1) % target_network_update_freq == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())

        print(f"Episode {e+1}/{num_episodes} completed.")

    agent.target_network.load_state_dict(agent.q_network.state_dict())

def evaluate(agent, data):
    state = data[0]
    total_reward = 0
    for t in range(1, len(data)):
        action = agent.act(state)
        next_state = data[t]
        reward = next_state[-2] - state[-2]  # Calculate reward based on the difference between the closing prices
        state = next_state
        total_reward += reward

    return total_reward

