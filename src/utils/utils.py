import numpy as np
from collections import namedtuple
from collections import deque
from utils.motion import Action
import random
import torch


MemoryUnit = namedtuple(
    "MemoryUnit", ["state", "action", "reward", "next_state", "done"]
)


class ReplayMemory(object):
    def __init__(self, memory_size) -> None:
        self.memory = deque([], maxlen=memory_size)

    def save(
        self,
        state: np.ndarray,
        action: Action,
        reward: int,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action.value, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.long)

        self.memory.append(
            MemoryUnit(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )
        )

    def sample(self, batch_size: int = 64) -> MemoryUnit:
        memories = (
            random.sample(self.memory, batch_size)
            if batch_size < len(self.memory)
            else self.memory
        )

        state, action, reward, next_state, done = zip(*memories)

        batch_state = torch.stack(state)
        batch_action = torch.stack(action)
        batch_reward = torch.stack(reward)
        batch_next_state = torch.stack(next_state)
        batch_done = torch.stack(done)

        return MemoryUnit(
            state=batch_state,
            action=batch_action,
            reward=batch_reward,
            next_state=batch_next_state,
            done=batch_done,
        )

    def __len__(self):
        return len(self.memory)
