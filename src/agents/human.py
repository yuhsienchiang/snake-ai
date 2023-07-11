import numpy as np
from agents.agent import Agent
from game.game import SnakeGame
from utils.motion import Direction, Action


class Human(Agent):
    def __init__(self, game: SnakeGame) -> None:
        super(Human, self).__init__(game=game)

    def get_action(self, state) -> np.ndarray:
        action = Action.STRAIGHT  # default action is straight

        current_direction = self.game.snake.get_direction()
        next_direction = current_direction

        for event in self.game.pygame.event.get():
            if event.type == self.game.pygame.QUIT:
                return Action.QUIT

            if event.type == self.game.pygame.KEYDOWN:
                if event.key == self.game.pygame.K_LEFT:
                    next_direction = Direction.LEFT
                elif event.key == self.game.pygame.K_RIGHT:
                    next_direction = Direction.RIGHT
                elif event.key == self.game.pygame.K_UP:
                    next_direction = Direction.UP
                elif event.key == self.game.pygame.K_DOWN:
                    next_direction = Direction.DOWN
                elif event.key == self.game.pygame.K_q:
                    return Action.QUIT

                d_idx = next_direction.value - current_direction.value

                if d_idx == 1 or d_idx == -3:
                    action = Action.RIGHT
                elif d_idx == -1 or d_idx == 3:
                    action = Action.LEFT

        return action
