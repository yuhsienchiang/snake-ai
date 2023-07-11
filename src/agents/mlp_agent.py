import numpy as np
import torch
from utils.motion import Action, Direction
from utils.ui import Point, BLOCK_SIZE
from game.game import SnakeGame
from agents.agent import Agent
from models.mlp import MLP_QNet


class MLPAgent(Agent):
    def __init__(self, game: SnakeGame) -> None:
        super(MLPAgent, self).__init__(game=game)
        self.main_net = MLP_QNet(11, 20, 3)
        self.target_net = MLP_QNet(11, 20, 3)
        self.target_net.load_state_dict(self.main_net.state_dict())

    def get_state(self) -> np.ndarray:
        snake_head = self.game.snake.get_head()

        point_l = Point(snake_head.x - BLOCK_SIZE, snake_head.y)
        point_r = Point(snake_head.x + BLOCK_SIZE, snake_head.y)
        point_u = Point(snake_head.x, snake_head.y - BLOCK_SIZE)
        point_d = Point(snake_head.x, snake_head.y + BLOCK_SIZE)

        dir_l = self.game.snake.get_direction() == Direction.LEFT
        dir_r = self.game.snake.get_direction() == Direction.RIGHT
        dir_u = self.game.snake.get_direction() == Direction.UP
        dir_d = self.game.snake.get_direction() == Direction.DOWN

        state = [
            # danger straight
            (dir_r and self.game.snake.is_collision(point_r))
            or (dir_l and self.game.snake.is_collision(point_l))
            or (dir_u and self.game.snake.is_collision(point_u))
            or (dir_d and self.game.snake.is_collision(point_d)),
            # danger right
            (dir_u and self.game.snake.is_collision(point_r))
            or (dir_d and self.game.snake.is_collision(point_l))
            or (dir_l and self.game.snake.is_collision(point_u))
            or (dir_r and self.game.snake.is_collision(point_d)),
            # danger left
            (dir_d and self.game.snake.is_collision(point_r))
            or (dir_u and self.game.snake.is_collision(point_l))
            or (dir_r and self.game.snake.is_collision(point_u))
            or (dir_l and self.game.snake.is_collision(point_d)),
            # move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # food location
            self.game.food.x < snake_head.x,
            self.game.food.x > snake_head.x,
            self.game.food.y < snake_head.y,
            self.game.food.y > snake_head.y,
        ]

        return np.array(state, dtype=np.int64)

    def get_action(self, state: np.ndarray) -> Action:
        action = [0, 0, 0]

        if state is None:
            idx_action = np.random.randint(0, 3)
        else:
            q_values = self.model(state)
            idx_action = torch.argmax(q_values).item()

        action[idx_action] = 1

        return Action(action)
