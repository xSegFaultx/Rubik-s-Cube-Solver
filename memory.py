from collections import deque
import numpy as np
import torch


class Memory:
    def __init__(self, max_length=3000):
        self.x = deque(maxlen=max_length)
        self.x_next = deque(maxlen=max_length)
        self.actions = deque(maxlen=max_length)
        self.rewards = deque(maxlen=max_length)
        self.is_terminals = deque(maxlen=max_length)

    def record(self, state, action, reward, next_state, is_terminal):
        self.x.append(state)
        self.x_next.append(next_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)

    def get_batch(self, batch_size):
        picked_index = np.random.choice(self.get_length(), batch_size, replace=False)
        x = np.array(list(self.x))[picked_index]
        x_next = np.array(list(self.x_next))[picked_index]
        action = np.array(list(self.actions))[picked_index]
        reward = np.array(list(self.rewards))[picked_index]
        is_terminal = np.array(list(self.is_terminals))[picked_index]
        return torch.tensor(x, dtype=torch.float), torch.tensor(x_next, dtype=torch.float), torch.tensor(action,
                                                                                                         dtype=torch.long), torch.tensor(
            reward, dtype=torch.float), torch.tensor(is_terminal, dtype=torch.bool)

    def get_length(self):
        return len(self.x)
