import numpy as np
from game.game import SnakeGame
from utils.motion import Action

class Agent(object):

    def __init__(self, game: SnakeGame) -> None:
        self.game = SnakeGame

    def get_state(self) -> np.ndarray:
        pass

    def get_action(self, state: np.ndarray) -> Action:
        pass
        
