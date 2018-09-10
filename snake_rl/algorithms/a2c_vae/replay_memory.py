import numpy as np


class ReplayMemory:
    def __init__(self):
        self.memory = []
        self.store_max_experiences = 200000

    def len_memory(self):
        return len(self.memory)

    def remember(self, experiences):
        self.memory.extend(experiences)
        self._forget_something()

    def _forget_something(self):
        while len(self.memory) > self.store_max_experiences:
            num_to_forget = int(0.05 * len(self.memory))
            self.memory = self.memory[num_to_forget:]
            np.random.shuffle(self.memory)

    def recollect(self, batch_size):
        idx = np.random.randint(len(self.memory) - batch_size)
        return self.memory[idx:idx + batch_size]
