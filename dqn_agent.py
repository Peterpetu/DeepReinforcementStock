import numpy as np
import torch
from model import QNetwork
import random

class DQNAgent:
    def __init__(self, state_size, action_size, device, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 memory_size=2000, batch_size=32, gamma=0.95, learning_rate=0.020):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.memory = []
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_network = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=False):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.tensor([state], dtype=torch.float, device=self.device)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state_tensor)
        self.q_network.train()
        return np.argmax(action_values.cpu().data.numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        state_tensor = torch.tensor([transition[0] for transition in minibatch], dtype=torch.float, device=self.device)
        action_tensor = torch.tensor([transition[1] for transition in minibatch], dtype=torch.long, device=self.device)
        reward_tensor = torch.tensor([transition[2] for transition in minibatch], dtype=torch.float, device=self.device)
        next_state_tensor = torch.tensor([transition[3] for transition in minibatch], dtype=torch.float, device=self.device)
        done_tensor = torch.tensor([transition[4] for transition in minibatch], dtype=torch.float, device=self.device)

        q_values = self.q_network(state_tensor)
        next_q_values = self.target_network(next_state_tensor).detach()

        q_values = q_values.gather(1, action_tensor.view(-1, 1)).squeeze()
        max_next_q_values, _ = next_q_values.max(1)
        target = reward_tensor + (1 - done_tensor) * self.gamma * max_next_q_values

        loss = torch.nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
