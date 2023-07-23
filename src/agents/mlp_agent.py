import numpy as np
import torch
from utils.motion import Action
from game.game import SnakeGame
from agents.agent import Agent
from models.mlp import MLP_QNet


class MLPAgent(Agent):
    def __init__(self, game: SnakeGame) -> None:
        super(MLPAgent, self).__init__(game=game)
        self.main_net = MLP_QNet([(11, 96), (96, 96), (96, 3)])
        self.target_net = MLP_QNet([(11, 96), (96, 96), (96, 3)])
        self.target_net.load_state_dict(self.main_net.state_dict())

    def get_action(self, state: np.ndarray) -> Action:
        action = [0, 0, 0]

        if state is None:
            idx_action = np.random.randint(0, 3)
        else:
            state = (
                state
                if isinstance(state, torch.Tensor)
                else torch.tensor(state, dtype=torch.float32)
            )

            # handdle input without batch dim
            if len(state.shape) == 2:
                state = torch.unsqueeze(state, 0)

            q_values = self.main_net(state)
            idx_action = torch.argmax(q_values).item()

        action[idx_action] = 1

        return Action(action)
