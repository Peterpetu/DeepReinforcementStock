import numpy as np
from environment import Environment
from dqn_agent import DQNAgent
import torch as torch

def main():
    # Load the preprocessed data
    train_data = np.load('ibm.us_train.npy')
    test_data = np.load('ibm.us_test.npy')

    # Create the environment
    env = Environment(train_data)

    # Create the agent
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size, seed=0, hidden_layers=[64, 64])

    # Number of episodes to train for
    n_episodes = 1000

    # Initialize the list to store total rewards per episode
    total_rewards = []

    for i_episode in range(1, n_episodes+1):
        # Reset the environment and the agent
        state = env.reset()

        done = False
        total_reward = 0
        step_counter = 0
        while not done:
            # Agent takes action
            action = agent.act(state)
            step_counter += 1
            if step_counter % 10 == 0:
                print(f"Current state: {state}, Action taken: {action}")

            # Get the next state, reward, and done from the environment
            next_state, reward, done, _ = env.step(action)
            # Agent learns from experience and updates Q-Network
            agent.step(state, action, reward, next_state, done)

            # Update the current state
            state = next_state

            total_reward += reward
            if step_counter % 10 == 0:
                print(f"Total reward so far: {total_reward}")

        # Append total reward of this episode to the list
        total_rewards.append(total_reward)

        # Print out some information about the training process
        print(f"Episode {i_episode}/{n_episodes} finished. Total reward: {total_reward}")

    # Save the trained Q-Network
    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')

    # Evaluate the agent's performance
    print("Evaluating agent's performance...")
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward
    print(f"Total reward: {total_reward}")

    # Save the total rewards per episode to a file
    np.save('total_rewards.npy', total_rewards)

if __name__ == "__main__":
    main()
