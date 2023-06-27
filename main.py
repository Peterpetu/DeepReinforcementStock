import numpy as np
from environment import StockTradingEnvironment
from dqn_agent import DQNAgent
import torch as torch

def main():
    # Load preprocessed data
    data = np.load('preprocessed_data.npy')

    # Create environment
    env = StockTradingEnvironment(data)

    # Create agent
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size, seed=0)

    # Number of episodes to train for
    n_episodes = 1000

    for i_episode in range(1, n_episodes+1):
        # Reset the environment and the agent
        state = env.reset()
        agent.reset()

        done = False
        while not done:
            # Agent takes action
            action = agent.act(state)

            # Get the next state, reward, and done from the environment
            next_state, reward, done = env.step(action)

            # Agent learns from experience and updates Q-Network
            agent.step(state, action, reward, next_state, done)

            # Update the current state
            state = next_state

        # Print out some information about the training process
        print(f"Episode {i_episode}/{n_episodes} finished.")

    # Save the trained Q-Network
    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')

if __name__ == "__main__":
    main()
