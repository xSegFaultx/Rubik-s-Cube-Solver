from memory import Memory
from model import Model
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from cube import Cube
import pickle


class Agent:
    def __init__(self):
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.decay = 0.9999
        self.action_space = 12
        self.gamma = 0.95
        self.episodes = 2000
        self.memory_size = 20000
        self.cube = Cube()

        self.train_threshold = 2048
        self.memory = Memory(self.memory_size)
        self.model = Model()
        self.model.load_state_dict(torch.load("./model_weights_1.dat"))
        self.model_p = Model()
        self.model_p.load_state_dict(torch.load("./model_weights_1.dat"))

        self.batch_size = 2048
        self.criterion = nn.MSELoss()
        self.lr = 0.0005
        self.optimizer = optim.Adam(params=self.model_p.parameters(), lr=self.lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)
        self.model_p = self.model_p.to(self.device)

    def act(self, state):
        explore_frame = np.random.rand()
        if explore_frame > self.epsilon:
            # exploit
            self.model_p.eval()
            state = np.array([state])
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            with torch.no_grad():
                Vs = self.model_p(state)
            self.model_p.train()
            return torch.argmax(Vs).item()
        else:
            # explore
            return np.random.randint(self.action_space)

    def replay(self):
        if self.memory.get_length() < self.train_threshold:
            return
        state, next_states, _, rewards, terminal_states = self.memory.get_batch(self.batch_size)
        next_states = torch.reshape(next_states, (-1, 20, 24))
        rewards = torch.flatten(rewards)
        terminal_states = (torch.flatten(terminal_states).int()-1)*-1
        state, next_states, rewards, terminal_states = state.to(self.device), next_states.to(self.device), rewards.to(self.device), \
                                                       terminal_states.to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_hat_next = self.model(next_states)
        self.model.train()
        y_hat_next = torch.max(y_hat_next, dim=1).values
        y = self.gamma*y_hat_next*terminal_states + rewards
        #print(y)
        y = torch.reshape(y, (-1, 12))
        self.model_p.zero_grad()
        y_hat = self.model_p(state)
        #print(y_hat)
        loss = torch.sqrt(self.criterion(y_hat, y))
        loss.backward()
        self.optimizer.step()
        print(loss.item())
        return loss.item()

    def train(self):
        total_rewards = []
        losses = []
        solved_counts = []
        for episode in range(self.episodes):
            self.cube.reset()
            self.cube.scramble(6)
            state = self.cube.get_state()
            is_terminal = False
            num_state = 0
            total_reward = 0
            while not is_terminal:
                action = self.act(state)
                curren_state, rewards, terminal_states, next_states, target_reward, target_terminal, target_next, target_solved = self.cube.peek(action)
                total_reward += target_reward
                self.memory.record(curren_state, action, rewards, next_states, terminal_states)
                if self.memory.get_length() > self.train_threshold:
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.decay
                is_terminal = target_terminal
                state = target_next
                num_state += 1
                if is_terminal:
                    print(
                        "episode: {}/{}, turns: {}, reward: {}, e: {:.2}, solved: {}".format(episode + 1, self.episodes, num_state,
                                                                                 total_reward, self.epsilon, target_solved))
                    loss = self.replay()
                    total_rewards.append(total_reward)
                    losses.append(loss)
                    if target_solved:
                        solved_counts.append(episode)
        with open("record_16.pkl", "wb") as record_file:
            pickle.dump([total_rewards, losses, solved_counts], record_file, protocol=pickle.HIGHEST_PROTOCOL)

    def test(self):
        self.model.load_state_dict(torch.load("./model_weights_1.dat"))
        done = False
        state = self.cube.get_state()
        state = np.array([state])
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        counts = 0
        while not done:
            self.model.eval()
            with torch.no_grad():
                y_hat = self.model(state)
            best_action = torch.max(y_hat, dim=1).indices[0]
            _, _, terminal_state, next_state, solved = self.cube.step(best_action)
            counts += 1
            if terminal_state:
                done = True
                # print("Steps: {} Solved: {}".format(counts, solved))
            next_state = np.array([next_state])
            next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
            state = next_state
        return solved

    def save(self, name):
        torch.save(self.model.state_dict(), "scramble_{}.dat".format(name))


if __name__ == "__main__":
    agent = Agent()
    agent.train()
