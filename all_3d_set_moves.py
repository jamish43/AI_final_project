# all_3d_path_plot.py

import numpy as np
from collections import defaultdict
import heapq
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# === 3D Environment Utilities ===

def mat3graph(grid):
    """
    Convert a 3D boolean occupancy grid into an adjacency list.
    grid: numpy array of shape (X,Y,Z), True=free, False=obstacle
    returns: dict mapping (x,y,z) to list of neighbor tuples
    """
    X, Y, Z = grid.shape
    graph = defaultdict(list)
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if not grid[x, y, z]:
                    continue
                for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    nx, ny, nz = x+dx, y+dy, z+dz
                    if 0 <= nx < X and 0 <= ny < Y and 0 <= nz < Z and grid[nx, ny, nz]:
                        graph[(x, y, z)].append((nx, ny, nz))
    return graph

def heuristic_euclidean(node, goal):
    return np.linalg.norm(np.array(node) - np.array(goal))

def heuristic_manhattan(node, goal):
    return sum(abs(a-b) for a, b in zip(node, goal))

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# === 1) A* path ===

def A_star_path_3d(graph, start, goal, heuristic=heuristic_euclidean):
    open_set = {start}
    came_from = {}
    gscore = defaultdict(lambda: float('inf')); gscore[start] = 0
    fscore = defaultdict(lambda: float('inf')); fscore[start] = heuristic(start, goal)
    while open_set:
        current = min(open_set, key=lambda n: fscore[n])
        if current == goal:
            return reconstruct_path(came_from, current)
        open_set.remove(current)
        for nbr in graph[current]:
            tg = gscore[current] + 1
            if tg < gscore[nbr]:
                came_from[nbr] = current
                gscore[nbr] = tg
                fscore[nbr] = tg + heuristic(nbr, goal)
                open_set.add(nbr)
    return []

# === 2) MA* path ===

def MA_star_path_3d(graph, start, goal, memory_limit, heuristic=heuristic_euclidean):
    open_set = {start}
    came_from = {}
    gscore = defaultdict(lambda: float('inf')); gscore[start] = 0
    fscore = defaultdict(lambda: float('inf')); fscore[start] = heuristic(start, goal)
    while open_set:
        current = min(open_set, key=lambda n: fscore[n])
        if current == goal:
            return reconstruct_path(came_from, current)
        open_set.remove(current)
        for nbr in graph[current]:
            tg = gscore[current] + 1
            if tg < gscore[nbr]:
                came_from[nbr] = current
                gscore[nbr] = tg
                fscore[nbr] = tg + heuristic(nbr, goal)
                open_set.add(nbr)
                if len(open_set) > memory_limit:
                    worst = max(open_set, key=lambda n: fscore[n])
                    open_set.remove(worst)
    return []

# === 3) SMA* path ===

sma_star_path_3d = MA_star_path_3d

# === 4) IDA* path ===

def IDA_star_path_3d(size, start, goal):
    dirs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    def h(n): return heuristic_manhattan(n, goal)

    def dfs(node, g, bound, visited):
        f = g + h(node)
        if f > bound:
            return None, f
        if node == goal:
            return [node], g
        min_bound = float('inf')
        for dx, dy, dz in dirs:
            nbr = (node[0]+dx, node[1]+dy, node[2]+dz)
            if any(c<0 or c>=size for c in nbr) or nbr in visited:
                continue
            visited.add(nbr)
            path, val = dfs(nbr, g+1, bound, visited)
            visited.remove(nbr)
            if path is not None:
                return [node] + path, val
            min_bound = min(min_bound, val)
        return None, min_bound

    bound = h(start)
    while True:
        path, val = dfs(start, 0, bound, {start})
        if path is not None:
            return path
        if val == float('inf'):
            return []
        bound = val

# === 5) Weighted A* path ===

def weighted_astar_path_3d(size, start, goal, weight):
    open_heap = [(weight * heuristic_manhattan(start, goal), start)]
    came_from = {}
    gscore = {start: 0}
    closed = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return reconstruct_path(came_from, current)
        closed.add(current)
        x, y, z = current
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nbr = (x+dx, y+dy, z+dz)
            if not all(0 <= c < size for c in nbr):
                continue
            tg = gscore[current] + 1
            if tg < gscore.get(nbr, float('inf')):
                came_from[nbr] = current
                gscore[nbr] = tg
                f = tg + weight * heuristic_manhattan(nbr, goal)
                heapq.heappush(open_heap, (f, nbr))
    return []

# === Plotting ===

def plot_paths_3d(paths, start, goal, size):
    """
    paths: dict of {algorithm_name: list of (x,y,z) tuples}
    start, goal: 3-tuples
    size: grid dimension for axis limits
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for name, path in paths.items():
        if not path:
            continue
        xs, ys, zs = zip(*path)
        ax.plot(xs, ys, zs, marker='o', label=name)

    ax.scatter([start[0]], [start[1]], [start[2]], marker='s', s=60, label='Start')
    ax.scatter([goal[0]], [goal[1]], [goal[2]], marker='*', s=100, label='Goal')

    ax.set_xlim(0, size-1)
    ax.set_ylim(0, size-1)
    ax.set_zlim(0, size-1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.show()

# === Main ===

if __name__ == '__main__':
    size = 10
    grid = np.ones((size, size, size), dtype=bool)
    graph = mat3graph(grid)
    free = list(graph.keys())
    start = (0, 0, 0)
    goal = random.choice(free)

    print(f"Start: {start}  Goal: {goal}")

    memory_limit = int(len(graph) * 0.5)
    weights = [1.0, 1.5, 2.0, 2.5]

    paths = {
        'A*': A_star_path_3d(graph, start, goal),
        'MA*': MA_star_path_3d(graph, start, goal, memory_limit),
        'SMA*': sma_star_path_3d(graph, start, goal, memory_limit),
        'IDA*': IDA_star_path_3d(size, start, goal),
    }

    # add weighted A* variants
    for w in weights:
        paths[f'wA* w={w}'] = weighted_astar_path_3d(size, start, goal, w)

    # plot all
    plot_paths_3d(paths, start, goal, size)
