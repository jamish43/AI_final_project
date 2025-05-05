import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import random
import pandas as pd

# Reconstruct path from came_from map
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# Euclidean heuristic function
def heuristic(cell, goal):
    return np.hypot(goal[0] - cell[0], goal[1] - cell[1])

# Memory-bounded A* search returns performance metrics (no visualization)
def memory_bounded_A_star_stats(graph, start, goal, memory_limit):
    t0 = time.time()
    open_set = {start}
    closed_set = set()
    came_from = {}
    gscore = {node: float('inf') for node in graph}
    gscore[start] = 0.0
    fscore = {node: float('inf') for node in graph}
    fscore[start] = heuristic(start, goal)

    nodes_explored = 0
    max_mem_usage = 0
    while open_set:
        nodes_explored += 1
        current = min(open_set, key=lambda n: fscore[n])
        if current == goal:
            break
        open_set.remove(current)
        closed_set.add(current)
        mem_use = len(open_set) + len(closed_set)
        max_mem_usage = max(max_mem_usage, mem_use)
        for neighbor in graph[current]:
            if neighbor in closed_set:
                continue
            tentative = gscore[current] + 1
            if tentative < gscore[neighbor]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative
                fscore[neighbor] = tentative + heuristic(neighbor, goal)
                open_set.add(neighbor)
                if len(open_set) + len(closed_set) > memory_limit:
                    worst = max(open_set, key=lambda n: fscore[n])
                    open_set.remove(worst)
    exec_time = time.time() - t0
    path = reconstruct_path(came_from, current)
    return exec_time, nodes_explored, max_mem_usage, len(path)

# Convert binary maze to adjacency graph
def mat2graph(mat):
    rows, cols = mat.shape
    graph = defaultdict(list)
    for x in range(rows):
        for y in range(cols):
            if mat[x, y]:
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and mat[nx, ny]:
                        graph[(x, y)].append((nx, ny))
    return graph

if __name__ == "__main__":
    from maze_gen_prims import PrimsMaze

    # Simulation parameters
    size_min, size_max = 2, 50
    runs_per_size = 1000
    start = (0, 0)
    memory_fraction = 0.5

    sizes = list(range(size_min, size_max + 1))
    avg_times, avg_nodes, avg_memory, avg_path = [], [], [], []

    print(f"Running memory-bounded A* ({runs_per_size} trials) for sizes {size_min} to {size_max}...")
    for idx, size in enumerate(sizes, 1):
        times_list, nodes_list, mem_list, path_list = [], [], [], []
        maze = PrimsMaze(size).create_maze(start)
        graph = mat2graph(maze)
        total_nodes = len(graph)
        memory_limit = max(10, int(total_nodes * memory_fraction))
        free_cells = list(graph.keys())

        for _ in range(runs_per_size):
            goal = random.choice(free_cells)
            t, ne, mu, pl = memory_bounded_A_star_stats(graph, start, goal, memory_limit)
            times_list.append(t)
            nodes_list.append(ne)
            mem_list.append(mu)
            path_list.append(pl)

        avg_times.append(np.mean(times_list))
        avg_nodes.append(np.mean(nodes_list))
        avg_memory.append(np.mean(mem_list))
        avg_path.append(np.mean(path_list))
        print(f"Size {size} [{idx}/{len(sizes)}] done (mem limit={memory_limit})", end='\r', flush=True)
    print()

    # Create table with pandas
    df = pd.DataFrame({
        'Maze Size (N x N)': sizes,
        'Avg Time (s)': avg_times,
        'Avg Nodes Explored': avg_nodes,
        'Avg Memory Used': avg_memory,
        'Avg Path Length': avg_path
    })
    print("\n===== Memory-Bounded A* Results =====")
    print(df.to_string(index=False))

    # Plotting metrics
    plt.figure(figsize=(12, 6))
    plt.plot(sizes, avg_nodes, marker='s', label='Avg Nodes Explored')
    plt.plot(sizes, avg_memory, marker='^', label='Avg Memory Used')
    plt.plot(sizes, avg_path, marker='d', label='Avg Path Length')
    plt.title(f"Memory-Bounded A* Metrics vs Size ({runs_per_size} trials)")
    plt.xlabel('Maze Size (N x N)')
    plt.ylabel('Average Metric')
    plt.xticks(np.arange(size_min, size_max + 1, 4))
    plt.legend(); plt.grid(True); plt.tight_layout()

    plt.figure(figsize=(10, 5))
    plt.plot(sizes, avg_times, marker='o', color='tab:orange', label='Avg Execution Time (s)')
    plt.title(f"Memory-Bounded A* Avg Execution Time vs Size ({runs_per_size} trials)")
    plt.xlabel('Maze Size (N x N)')
    plt.ylabel('Average Time (s)')
    plt.xticks(np.arange(size_min, size_max + 1, 4))
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()
