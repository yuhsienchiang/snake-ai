from enum import Enum

SPEED = 20

class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

class Action(Enum):
    STRAIGHT = [1, 0, 0]
    RIGHT = [0, 1, 0]
    LEFT = [0, 0, 1]
    QUIT = [0, 0, 0]
