import numpy as np
from environment import Environment

def load_test_data():
    # Load the testing data
    test_data = np.load('ibm.us_test.npy')
    return test_data

def create_environment(test_data):
    # Create the environment with the testing data
    env = Environment(test_data)
    return env

if __name__ == "__main__":
    test_data = load_test_data()
    env = create_environment(test_data)
    print("Test data loaded and environment created.")
