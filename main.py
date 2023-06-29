import numpy as np
from environment import Environment
from dqn_agent import DQNAgent
import torch as torch
import logging

logging.basicConfig(level=logging.INFO)

def main():
    # Load the preprocessed data
    train_data = np.load('ibm.us_train.npy')
    test_data = np.load('ibm.us_test.npy')

    # Create the environment
    env = Environment(train_data)
    total_steps = len(train_data)
    steps_taken = 0
    print(f"Starting new episode with {total_steps} steps.")


    

    # Create the agent
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size, seed=0, hidden_layers=[64, 64])

    # Number of episodes to train for
    n_episodes = 1000

    # Initialize the list to store total rewards per episode
    total_rewards = []

    # Epsilon parameters
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    eps = eps_start

    for i_episode in range(1, n_episodes+1):
        # Reset the environment and the agent
        state = env.reset()

        done = False
        total_reward = 0
        step_counter = 0

        while not done:
            action = agent.act(state, eps)
            steps_taken += 1
            #if steps_taken >= 10000:
            #    print("10000 steps taken, stopping execution.")
            #    return
            step_counter += 1
            print(step_counter)
            if step_counter % 1000 == 0:  # Only print every 1000 steps
                logging.info(f"Step: {step_counter}, Action taken: {action}")
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if i_episode % 10 == 0:  # Only print every 10 episodes
                logging.info(f"Episode: {i_episode}, Total reward: {total_reward}")

        # Append total reward of this episode to the list
        total_rewards.append(total_reward)

        # Decrease epsilon
        eps = max(eps_end, eps_decay*eps)

        # Print out some information about the training process
        logging.info(f"Episode: {i_episode}, Total reward: {total_reward}, Epsilon: {eps}")

    # Save the trained Q-Network
    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    logging.info("Model saved.")

    # Evaluate the agent's performance
    logging.info("Evaluating agent's performance...")
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
    logging.info(f"Total reward: {total_reward}")

    # Save the total rewards per episode to a file
    np.save('total_rewards.npy', total_rewards)

if __name__ == "__main__":
    main()
