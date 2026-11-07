"""
Mathematical Foundations of Deep Q-Networks (DQN)

1. Markov Decision Process (MDP):
   - Defined by the tuple (S, A, P, R, gamma).
   - The optimal action-value function Q*(s, a) satisfies the Bellman Optimality Equation:
     Q*(s, a) = E [ R_{t+1} + gamma * max_{a'} Q*(S_{t+1}, a') | S_t = s, A_t = a ]

2. Non-linear Function Approximation:
   - For continuous or high-dimensional state spaces (s in R^d), tabular Q-learning is infeasible.
   - Deep neural network with parameters theta approximates the Q-function: Q(s, a; theta).

3. Mean Squared Bellman Error (MSBE):
   - Measures the discrepancy between current predictions and the target label:
     L(theta) = E_{(s, a, r, s', d) ~ D} [ ( r + gamma * (1 - d) * max_{a'} Q(s', a'; theta^-) - Q(s, a; theta) )^2 ]

4. Bootstrapping & Moving Target Problem:
   - Calculating target y = r + gamma * max_{a'} Q(s', a'; theta) using the active model itself is "bootstrapping".
   - If theta = theta^-, the target moves every optimization step, causing gradient divergence.
   - Solution: Decouple the target network (theta^-) and apply Polyak soft updates (tau << 1):
     theta^- <- tau * theta + (1 - tau) * theta^-
"""

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# 1. Action-Value Function Approximator
# Given a state vector s in R^d, outputs Q-values for all discrete actions in A.
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            # Projects lower-dimensional state representation into higher-dimensional subspace: h1 = max(0, s W1^T + b1)
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # Output layer: Keeps linear activation without ReLU/Softmax.
            # Cumulative discounted rewards range over the entire real line (-inf to +inf).
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# 2. Experience Replay Buffer
# Destroys temporal autocorrelation across sequential transitions (Cov(S_t, S_{t+1}) >> 0).
# Ensures independent and identically distributed (i.i.d.) conditions for SGD by sampling uniformly at random.
class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        # Circular double-ended queue: evicts the oldest transition in O(1) when capacity is exceeded.
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        # Uniform random sampling without replacement from buffer D: (s, a, r, s', d) ~ U(D)
        states, actions, rewards, next_states, dones = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),  # actions serve as 64-bit integer channel indices for .gather()
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


# 3. DQN Agent (Policy Evaluation & Optimization Engine)
class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,  # Polyak averaging coefficient (tracks 0.5% per step)
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)

        # Decouple networks to eliminate the Moving Target problem:
        # q_net (active weights theta) vs. target_net (quasi-static weights theta^-)
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())  # Initialize with identical parameters

        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()

    def select_action(self, state, epsilon: float = 0.05) -> int:
        # Epsilon-Greedy Policy: Exploration vs. Exploitation
        if random.random() < epsilon:
            # Exploration: Sample uniformly from discrete action space to avoid local optima
            return random.randint(0, self.action_dim - 1)

        # Exploitation: Greedy choice a* = argmax_a Q(s, a; theta)
        # unsqueeze(0) expands 1D state vector (state_dim,) into 2D batch tensor (1, state_dim)
        state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def update(self, batch_size: int = 64):
        # Prevent training until the replay buffer has accumulated enough transitions
        if len(self.memory) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 1. Extract Q(s, a) for the actual actions executed
        # actions.unsqueeze(1) transforms shape from (batch_size,) to (batch_size, 1) [2D column vector]
        # .gather(1, index) selects the corresponding Q-value along dimension 1
        # .squeeze(1) flattens the result back to (batch_size,)
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 2. Compute Bellman Target Label using the Target Network:
        # y = r + gamma * (1 - done) * max_{a'} Q(s', a'; theta^-)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            # (1.0 - dones): terminal states have zero future discounted returns, leaving target_q = rewards
            target_q = rewards + (1.0 - dones) * self.gamma * max_next_q

        # 3. Huber Loss (Smooth L1 Loss) & Backpropagation
        # Smooth L1 combines L2 smoothness for small errors (|e| <= 1) and L1 linearity for large TD-errors (|e| > 1)
        # Prevents exploding gradients when anomalous rewards or high TD errors occur.
        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping: Clamps L2-norm of gradients to max_norm=1.0 to prevent gradient explosion
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 4. Polyak Soft Update (Exponential Moving Average tracking)
        # Replaces periodic hard updates with smooth, continuous parameter tracking:
        # theta^- <- tau * theta + (1 - tau) * theta^-
        for param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        return loss.item()