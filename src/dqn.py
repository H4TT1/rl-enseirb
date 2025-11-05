import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from src.base_agent import BaseAgent

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    def forward(self, state):
        return self.net(state)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, *args):
        max_prio = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
        else:
            self.memory[self.pos] = Transition(*args)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        return samples, indices, torch.tensor(weights)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)

class DQNAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, buffer_size=10000, batch_size=64, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.update_target_net()
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(buffer_size, alpha)

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def select_action(self, state, greedy=False):
        if not greedy and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            return self.policy_net(state).max(1)[1].view(1, 1).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        transitions, indices, weights = self.memory.sample(self.batch_size, beta)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat([torch.from_numpy(s).float().unsqueeze(0) for s in batch.state])
        action_batch = torch.tensor(batch.action).long().unsqueeze(1)
        reward_batch = torch.tensor(batch.reward).float()
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size)
        non_final_mask = torch.tensor([ns is not None for ns in batch.next_state], dtype=torch.bool)
        non_final_next_states = torch.cat([torch.from_numpy(ns).float().unsqueeze(0) for ns in batch.next_state if ns is not None])
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        errors = torch.abs(state_action_values.squeeze() - expected_state_action_values)
        loss = (torch.tensor(weights) * F.mse_loss(state_action_values.squeeze(), expected_state_action_values, reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.update_priorities(indices, errors.detach().numpy() + 1e-5)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))

def train(env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score):
    agent = DQNAgent(state_dim, action_dim)
    scores_deque = deque(maxlen=100)
    scores = []
    print("Starting DQN training...")
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_for_buffer = None if done else next_state
            agent.memory.push(state, action, reward, next_state_for_buffer, done)
            agent.learn()
            state = next_state
            episode_reward += reward
            if done:
                break
        scores_deque.append(episode_reward)
        scores.append(episode_reward)
        if episode % num_episodes == 0:
            agent.update_target_net()
        if episode % (num_episodes / 10) == 0:
            print(f"Episode {episode}\tAverage Score: {np.mean(scores_deque):.2f}")
        if np.mean(scores_deque) >= target_score:
            print(f"\nEnvironment solved in {episode} episodes! Average Score: {np.mean(scores_deque):.2f}")
            break
    print("\nTraining complete.")
    return agent, scores
