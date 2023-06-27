import pandas as pd
import torch
from dqn_agent import DQNAgent
from train_evaluate import train, evaluate


def main():
    # Replace with your own API key
    api_key = 'QEKC128R0MKLL6C7'
    # Call the fetch_stock_data function
    data = fetch_stock_data(api_key, 'GOOGL', output_file='googl_data.csv')

    first_half, second_half = split_data(data)

    first_half_preprocessed = preprocess_data(first_half)
    second_half_preprocessed = preprocess_data(second_half)
    save_preprocessed_data_to_csv(first_half_preprocessed, second_half_preprocessed)

    # Convert the preprocessed data to float32
    first_half_preprocessed = first_half_preprocessed.values.astype('float32')
    second_half_preprocessed = second_half_preprocessed.values.astype('float32')

    state_size = first_half_preprocessed.shape[1]
    action_size = 3  # Buy, Hold, Sell
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DQNAgent(state_size, action_size, device)
    torch.manual_seed(42)

    train(agent, first_half_preprocessed)
    reward = evaluate(agent, second_half_preprocessed)

    print(f"Total reward from evaluation: {reward}")

if __name__ == "__main__":
    main()