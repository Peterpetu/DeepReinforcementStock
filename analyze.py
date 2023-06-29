import numpy as np
import matplotlib.pyplot as plt

# Load the data
total_rewards = np.load('total_rewards.npy')

# Create a plot
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.show()

print(total_rewards)
print(total_rewards.shape)
