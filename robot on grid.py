import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# -------------------- Environment --------------------
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.obstacles = [(2,2), (3,3)]
        self.state = self.start
        self.steps = 0
        self.max_steps = 100

    def reset(self):
        self.state = self.start
        self.steps = 0
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:   # up
            x = max(0, x-1)
        elif action == 1: # down
            x = min(self.size-1, x+1)
        elif action == 2: # left
            y = max(0, y-1)
        elif action == 3: # right
            y = min(self.size-1, y+1)
        next_state = (x, y)
        self.steps += 1

        # Reward calculation
        if next_state == self.goal:
            reward = 10
            done = True
        elif next_state in self.obstacles:
            reward = -1
            done = True   # episode ends on obstacle
        elif self.steps >= self.max_steps:
            reward = -1
            done = True
        else:
            reward = -0.01   # small penalty per step
            done = False

        self.state = next_state
        return next_state, reward, done

# -------------------- Neural Network (Q-function) --------------------
class DQN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# -------------------- Agent (DQN) --------------------
class Agent:
    def __init__(self, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_network = DQN()
        self.target_network = DQN()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=2000)
        self.batch_size = 32

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randint(0, 3)
        state_t = torch.tensor(state, dtype=torch.float32)
        q_values = self.q_network(state_t)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# -------------------- Training Loop --------------------
env = GridWorld()
agent = Agent()
episodes = 300
target_update_freq = 20

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward
    if episode % target_update_freq == 0:
        agent.update_target()
    if episode % 50 == 0:
        print(f"Episode {episode}, Total reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

print("Training finished. Testing the learned policy...")

# -------------------- Test the trained agent --------------------
state = env.reset()
done = False
path = [state]
while not done:
    action = agent.act(state)   # epsilon still used, but epsilon is now very low
    next_state, reward, done = env.step(action)
    path.append(next_state)
    state = next_state
print("Path taken:", path)