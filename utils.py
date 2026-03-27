import numpy as np

class Node:
    def __init__(self, state):
        self.state = state
        self.P = None   # prior probabilities
        self.N = np.zeros(4672)  # how many times each action has been picked
        self.W = np.zeros(4672)  # cumulative value
        self.Q = np.zeros(4672)  # current Q-value for each action (mean cum value)
        self.children = {}
        self.is_expanded = False

class ReplayBuffer:
    def __init__(self, capacity, state_shape=(20, 8, 8), action_size=4672):
        self.capacity = capacity
        self.state_shape = state_shape
        self.action_size = action_size

        # Pre-allocate memory for efficiency
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.policies = np.zeros((capacity, action_size), dtype=np.float32)
        self.values = np.zeros((capacity, 1), dtype=np.float32)

        self.index = 0
        self.size = 0

    def add(self, state, policy, value):
        self.states[self.index] = state
        self.policies[self.index] = policy
        self.values[self.index] = value

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, states, policies, values):
        for s, p, v in zip(states, policies, values):
            self.add(s, p, v)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            self.states[indices],
            self.policies[indices],
            self.values[indices]
        )

    def __len__(self):
        return self.size