from collections import namedtuple
from enum import Enum

BLOCK_SIZE = 20
SPEED = 10
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

Point = namedtuple("Point", ["x", "y"])


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class Action(Enum):
    STRAIGHT = [1, 0, 0]
    RIGHT = [0, 1, 0]
    LEFT = [0, 0, 1]
