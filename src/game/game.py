import pygame
import numpy as np
import math
import random
from .snake import Snake
from utils.motion import SPEED
from utils.ui import Point, BLOCK_SIZE, RED, BLUE1, BLUE2, GREEN1, GREEN2, BLACK, WHITE


class SnakeGame(object):
    def __init__(self, width, height) -> None:
        self.pygame = pygame
        self.pygame.init()
        self.font = self.pygame.font.SysFont("FiraCode Nerd Font", size=25)
        self.window_width = width
        self.window_height = height

        self.grid_width = width // BLOCK_SIZE
        self.grid_height = height // BLOCK_SIZE
        self.grid_size = self.grid_height * self.grid_width

        self.window = self.pygame.display.set_mode(
            (self.window_width, self.window_height)
        )
        self.pygame.display.set_caption("Snake")
        self.clock = self.pygame.time.Clock()
        self.reset()

    def reset(self) -> np.ndarray:
        self.snake = Snake(
            head_x=(((self.window_width // BLOCK_SIZE) - 1) // 2) * BLOCK_SIZE,
            head_y=(((self.window_height // BLOCK_SIZE) - 1) // 2) * BLOCK_SIZE,
            boundary_x=self.window_width,
            boundary_y=self.window_height,
        )
        self._place_food()
        self.score = 0
        self.frame_iteration = 0

        return self.get_state()

    def get_state(self) -> np.ndarray:
        state = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)

        state.T[
            tuple((np.transpose(self.snake.body) - BLOCK_SIZE) // BLOCK_SIZE)
        ] = np.linspace(0.8, 0.2, len(self.snake))
        state.T[
            tuple((np.transpose(self.snake.get_head()) - BLOCK_SIZE) // BLOCK_SIZE)
        ] = 1.0
        state.T[tuple((np.transpose(self.food) - BLOCK_SIZE) // BLOCK_SIZE)] = -1.0

        return state

    def _place_food(self) -> None:
        x = (
            random.randint(
                0,
                (self.window_width - BLOCK_SIZE) // BLOCK_SIZE,
            )
            * BLOCK_SIZE
        )
        y = (
            random.randint(
                0,
                (self.window_height - BLOCK_SIZE) // BLOCK_SIZE,
            )
            * BLOCK_SIZE
        )
        self.food = Point(x, y)
        if self.snake.is_collision(pt=self.food):
            self._place_food()

    def play_step(self, action) -> tuple[np.ndarray, float, bool, int]:
        self.frame_iteration += 1

        self.snake.move(action)
        next_state = self.get_state()

        # check done
        if self.snake.is_collision() or self.frame_iteration > 100 * len(self.snake):
            # snake collide or idle
            done = True
        else:
            done = False

        # compute reward
        curr_head = self.snake.get_head(0)
        prev_head = self.snake.get_head(1)

        if done:
            reward = len(self.snake) - self.grid_size
        elif self.food == curr_head:
            # snkae get food
            reward = math.exp((self.grid_size - self.frame_iteration) / self.grid_size)

            self.score += 1
            self.frame_iteration = 0
            self._place_food()

        else:
            if self._distance(curr_head, self.food) < self._distance(prev_head, self.food):
                # 1. distance to food -> small
                # 2. iteration -> small
                # 3. snake length -> small
                # 4. move toward food +-
                reward = 1.0 * len(self.snake) / self._distance(curr_head, self.food) 
            else:
                reward = (
                    - self._distance(curr_head, self.food)  / (10.0 * len(self.snake))
                )

            self.snake.body.pop()
            next_state = self.get_state()

        # update ui
        if not done:
            self._update_ui()
            self.clock.tick(SPEED)

        return next_state, reward, done, self.score

    def _distance(self, point_1: Point, point_2: Point) -> float:
        return np.linalg.norm(np.subtract(point_1, point_2)) / BLOCK_SIZE

    def _update_ui(self) -> None:
        # fill the background
        self.window.fill(BLACK)

        # draw the snake
        for id, pt in enumerate(self.snake.body):
            core_color = BLUE2 if id != 0 else GREEN2
            border_color = BLUE1 if id != 0 else GREEN1

            self.pygame.draw.rect(
                self.window,
                border_color,
                self.pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE),
            )
            self.pygame.draw.rect(
                self.window,
                core_color,
                self.pygame.Rect(pt.x + 4, pt.y + 4, 12, 12),
            )

        # draw the food
        self.pygame.draw.rect(
            self.window,
            RED,
            self.pygame.Rect(
                self.food.x,
                self.food.y,
                BLOCK_SIZE,
                BLOCK_SIZE,
            ),
        )

        # print out the score
        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.window.blit(text, [4, 1])

        # update the frame
        self.pygame.display.flip()

    def end_game_ui(self, title, score) -> None:
        self.window.fill(BLACK)

        title_text = self.font.render(title, True, WHITE)
        title_text_rect = title_text.get_rect(
            center=(self.window_width / 2, self.window_height / 2)
        )
        self.window.blit(title_text, title_text_rect)

        score_text = self.font.render("Score: " + str(score), True, WHITE)
        score_text_rect = title_text_rect.copy().move(0, 30)
        self.window.blit(score_text, score_text_rect)

        self.pygame.display.flip()
