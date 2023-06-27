import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(csv_filename, window_size=10):
    # Load the data
    data = pd.read_csv(csv_filename)

    # Create additional features
    data['Range'] = data['High'] - data['Low']
    data['Return'] = data['Close'] - data['Open']

    # Drop the 'Date' and 'OpenInt' columns
    data = data.drop(['Date', 'OpenInt'], axis=1)

    # Normalize the data
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    # Create windows
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data.iloc[i:i+window_size].values)
    windows = np.array(windows)

    # Split into training and test sets
    split_index = int(len(windows) * 0.8)
    train_data = windows[:split_index]
    test_data = windows[split_index:]

    # Save the preprocessed data
    np.save(csv_filename.replace('.csv', '_train.npy'), train_data)
    np.save(csv_filename.replace('.csv', '_test.npy'), test_data)

    print(f"Preprocessed data saved to {csv_filename.replace('.csv', '_train.npy')} and {csv_filename.replace('.csv', '_test.npy')}")

# Usage
preprocess_data('ibm.us.csv')
