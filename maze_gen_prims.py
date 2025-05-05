import numpy as np
import random

class PrimsMaze:
    def __init__(self, size, display=False):
        self.size = size
        self.display = display
        self.maze = np.zeros((size, size), dtype=bool)

    def create_maze(self, start=(0,0)):
        walls = []
        self.maze[start] = True

        def add_walls(x, y):
            directions = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            for nx, ny in directions:
                if 0 <= nx < self.size and 0 <= ny < self.size and not self.maze[nx, ny]:
                    walls.append((nx, ny, x, y))

        add_walls(*start)

        while walls:
            idx = random.randint(0, len(walls)-1)
            x, y, px, py = walls.pop(idx)
            if not self.maze[x, y]:
                neighbors = [(nx, ny) for nx, ny in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
                             if 0 <= nx < self.size and 0 <= ny < self.size and self.maze[nx, ny]]
                if len(neighbors) == 1:
                    self.maze[x, y] = True
                    add_walls(x, y)

        return self.maze
