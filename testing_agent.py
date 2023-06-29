import numpy as np
from environment import Environment
from dqn_agent import DQNAgent
import torch

def load_test_data():
    # Load the testing data
    test_data = np.load('ibm.us_test.npy')
    return test_data

def create_environment(test_data):
    # Create the environment with the testing data
    env = Environment(test_data)
    return env

def create_agent(env):
    # Create the agent
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size, seed=0, hidden_layers=[64, 64])
    return agent

def load_trained_model(agent):
    # Load the trained model
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    print("Trained model loaded.")

def test_agent(env, agent):
    # Initialize the total rewards
    total_rewards = 0

    # Reset the environment
    state = env.reset()

    done = False

    # Run the testing loop
    while not done:
        # Select an action
        action = agent.act(state)

        # Print the current state and the selected action
        print('State:', state)
        print('Action:', action)

        # Take a step in the environment
        next_state, reward, done, _ = env.step(action)

        # Print the reward and the next state
        print('Reward:', reward)
        print('Next state:', next_state)

        # Update the total rewards
        total_rewards += reward

        # Move to the next state
        state = next_state

    # Print the total rewards
    print('Total rewards:', total_rewards)

if __name__ == "__main__":
    test_data = load_test_data()
    env = create_environment(test_data)
    agent = create_agent(env)
    load_trained_model(agent)
    test_agent(env, agent)
