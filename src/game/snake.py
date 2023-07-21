import copy
from utils.motion import Direction, Action
from utils.ui import BLOCK_SIZE, Point


class Snake(object):
    def __init__(self, head_x, head_y, boundary_x, boundary_y) -> None:
        self.boundary_x = boundary_x
        self.boundary_y = boundary_y
        self.direction = Direction.RIGHT
        self.body = [
            Point(head_x, head_y),
            Point(head_x - BLOCK_SIZE, head_y),
            Point(head_x - (2 * BLOCK_SIZE), head_y),
        ]

    def get_direction(self):
        return copy.deepcopy(self.direction)

    def set_direction(self, directionion):
        self.direction = directionion

    def get_head(self, prev_head=0):
        if prev_head >= len(self.body):
            return -1

        return copy.deepcopy(self.body[prev_head])

    def insert_head(self, head):
        self.body.insert(0, head)

    def __len__(self):
        return len(self.body)

    def is_collision(self, pt=None):
        pt = self.get_head() if pt is None else pt

        # hits wall
        if (
            pt.x > self.boundary_x - BLOCK_SIZE
            or pt.x < 0
            or pt.y > self.boundary_y - BLOCK_SIZE
            or pt.y < 0
        ):
            return True
        # hits itself
        if pt in self.body[1:]:
            return True
        else:
            return False

    def move(self, action):
        current_direction = self.get_direction()

        if action == Action.STRAIGHT:
            next_direction = current_direction
        elif action == Action.RIGHT:
            next_direction = Direction((current_direction.value + 1) % 4)
        elif action == Action.LEFT:
            next_direction = Direction((current_direction.value - 1) % 4)

        self.set_direction(next_direction)

        head = self.get_head(prev_head=0)
        head_x = head.x
        head_y = head.y

        if self.get_direction() == Direction.RIGHT:
            head_x += BLOCK_SIZE
        elif self.get_direction() == Direction.LEFT:
            head_x -= BLOCK_SIZE
        elif self.get_direction() == Direction.DOWN:
            head_y += BLOCK_SIZE
        elif self.get_direction() == Direction.UP:
            head_y -= BLOCK_SIZE

        self.insert_head(Point(head_x, head_y))
