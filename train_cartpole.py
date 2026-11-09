"""
Day 6: Deep Q-Network (DQN) Training on CartPole-v1

1. Epsilon-Greedy Strategy:
   - Initial Phase: Parameters are randomly initialized, so exploration is prioritized (ε ~ 1.0)
     to prevent premature convergence to local optima.
   - Exploitation Phase: As Q-values converge, the agent acts greedily (1 - ε), exploiting
     learned state-action value estimates.

2. Discount Factor (Gamma) & Effective Horizon:
   - Cumulative Return: G_t = ∑_{k=0}^∞ (γ^k * r_{t+k})
   - In CartPole, r_t = +1 for every step the pole remains upright.
   - For γ = 0.99, the effective planning horizon is:
     1 / (1 - γ) = 1 / (1 - 0.99) = 100 steps.
   - The agent considers long-term stability (>100 steps ahead) rather than just the immediate step.

3. CartPole-v1 State & Action Specifications:
   - State (Continuous 4D): [Cart Position (x), Cart Velocity (x_dot), 
                            Pole Angle (theta), Pole Angular Velocity (theta_dot)]
   - Action (Discrete 2D): 0 (Push Cart Left), 1 (Push Cart Right)
   - Termination: Pole angle > ±12° (~0.209 rad) or Cart position > ±2.4
   - Truncation: Maximum episode steps reached (500 steps in CartPole-v1)
   - Environment Solved: 100-episode moving average reward >= 475.0
"""

import gymnasium as gym
import numpy as np
import torch
from deepq import DQNAgent


def train_dqn():
    env = gym.make("CartPole-v1")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training DQN on CartPole-v1 using device: {device}")

    state_dim = env.observation_space.shape[0]  # Continuous 4D state vector
    action_dim = env.action_space.n             # Discrete action space (0: left, 1: right)

    # Hyperparameters
    max_episodes = 600
    batch_size = 64
    gamma = 0.99
    lr = 1e-3
    tau = 0.005

    # Epsilon decay schedule
    epsilon = 1.0
    min_epsilon = 0.01
    epsilon_decay = 0.985

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        tau=tau,
        device=device,
    )

    scores = []

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # Epsilon-Greedy Action Selection:
            # Select random action with probability ε, otherwise argmax_a Q(s, a; θ)
            action = agent.select_action(state, epsilon=epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition tuple (s, a, r, s', done) in Replay Buffer
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            # Gradient Step: Sample mini-batch once buffer has >= batch_size transitions.
            # Computes Huber loss on TD-error and performs Polyak soft target update.
            agent.update(batch_size=batch_size)

        # Decay epsilon after each episode
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        scores.append(episode_reward)

        # Compute moving average across the last 100 episodes (standard benchmark)
        avg_score_100 = np.mean(scores[-100:])
        avg_score_10 = np.mean(scores[-10:])

        if episode % 10 == 0:
            print(
                f"Episode {episode:3d} | "
                f"Last 10 Avg: {avg_score_10:6.2f} | "
                f"100 Avg: {avg_score_100:6.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )

        # CartPole-v1 solved condition: 100-episode moving average >= 475.0
        if len(scores) >= 100 and avg_score_100 >= 475.0:
            print(f"\n[Environment Solved!] Episode {episode} achieved 100-episode average reward of {avg_score_100:.2f}")
            break

    env.close()


if __name__ == "__main__":
    train_dqn()