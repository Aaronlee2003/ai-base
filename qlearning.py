#ql - Q-learning algorithm implementation
#q learning is a  maxization  future reward algorithm
# using bellman equation to update Q-values
# using temporal difference learning to update Q-values
#Weights of $(1-\alpha)$ are applied to the existing data, and weights of α are applied to the new experience, then the two are combined to produce a new expectation.
#epsilon-greedy policy is act look like stupid this is  solution greedy is trap in local maximum
import numpy as np

class SimpleGridWorld:
    def __init__(self, size=4):
        self.size = size
        self.num_states = size * size
        self.num_actions = 4
        self.pits = {5, 7, 11, 12}
        self.goal = 15
        self.state = 0
    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        row, col = divmod(self.state, self.size)# map state to row and column show the which  using mod and div operations

        # # Apply action: up, down, left, right
        if action == 0 and row > 0:
            row -= 1
        elif action == 1 and row < self.size - 1:
            row += 1
        elif action == 2 and col > 0:
            col -= 1
        elif action == 3 and col < self.size - 1:
            col += 1

        self.state = row * self.size + col #or convert row and column back to state index
        if self.state == self.goal:# check if reached goal state
            return self.state, 10.0, True
        elif self.state in self.pits:# check if fell into a pit
            return self.state, -10.0, True
        else:
            return self.state, -1.0, False
        
class QLearningAgent:
    def __init__(self, num_states: int, num_actions: int, lr: float = 0.1, gamma: float = 0.99, epsilon: float = 1.0, epsilon_decay: float = 0.995, min_epsilon: float = 0.01):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr                    # 학습률 (alpha)
        self.gamma = gamma              # discount factor for future rewards
        self.epsilon = epsilon          # exploration probability for epsilon-greedy policy
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros((num_states, num_actions))  # Initialize Q-table with zeros
        #state*action size Q-table for storing Q-values of each state-action pair
    def choose_action(self, state: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)  # Explore: random action
        else:
            return np.argmax(self.q_table[state])      # if random number is greater than epsilon, choose the best action based on Q-table
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """Update Q-table using the Bellman optimality equation"""
        current_q = self.q_table[state, action]

        if done:# if we have reached a terminal state, the target is just the reward
            target = reward
        else:
            # Extract the maximum Q-Value available in the future state next_state (key for Off-Policy)
            max_future_q = np.max(self.q_table[next_state])
            target = reward + self.gamma * max_future_q

        # Update Q-value based on TD Error
        td_error = target - current_q
        self.q_table[state, action] += self.lr * td_error

    def decay_epsilon(self):
        """Gradually decrease exploration rate as learning progresses"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay) # multiply epsilon by decay factor but ensure it doesn't go below min_epsilon
        
    def train_q_learning(self, episodes: int = 1000):
        env = SimpleGridWorld(size=4)
        agent = QLearningAgent(num_states=env.num_states, num_actions=env.num_actions)
        for ep in range(1, episodes + 1):#repeated episodes for training
         state = env.reset()#make initial state for the episode
         done = False
         total_reward = 0.0

         while not done:
            action = agent.choose_action(state) #choose action based on epsilon-greedy policy
            next_state, reward, done = env.step(action)#take action and observe next state and reward
            agent.update(state, action, reward, next_state, done)#update Q-table based on observed transition

            state = next_state
            total_reward += reward

         agent.decay_epsilon()#decay exploration rate after each episode
         if ep % 200 == 0:
            print(f"Episode {ep:4d} | Total Reward: {total_reward:6.1f} | Epsilon: {agent.epsilon:.4f}")

        print("\n최적 Q-Table 학습 완료!")
        return agent, env

    