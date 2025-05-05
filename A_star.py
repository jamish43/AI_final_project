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
    return np.hypot(goal[0] - cell[0], cell[1] - goal[1])

# A* search returns performance metrics (no visualization)
def A_star_stats(graph, start, goal):
    t0 = time.time()
    open_set = {start}
    closed_set = set()
    came_from = {}
    gscore = {node: float('inf') for node in graph}
    gscore[start] = 0.0
    fscore = {node: float('inf') for node in graph}
    fscore[start] = heuristic(start, goal)

    nodes_explored = 0
    max_space = 0
    while open_set:
        nodes_explored += 1
        current = min(open_set, key=lambda n: fscore[n])
        if current == goal:
            break
        open_set.remove(current)
        closed_set.add(current)
        max_space = max(max_space, len(open_set) + len(closed_set))
        for neighbor in graph[current]:
            if neighbor in closed_set:
                continue
            tentative = gscore[current] + 1
            if tentative < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative
                fscore[neighbor] = tentative + heuristic(neighbor, goal)
                open_set.add(neighbor)
    exec_time = time.time() - t0
    path = reconstruct_path(came_from, current)
    return exec_time, nodes_explored, max_space, len(path)

# Convert binary maze to adjacency graph
def mat2graph(mat):
    rows, cols = mat.shape
    graph = defaultdict(list)
    for x in range(rows):
        for y in range(cols):
            if mat[x, y]:
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < rows and 0 <= ny < cols and mat[nx, ny]:
                        graph[(x, y)].append((nx, ny))
    return graph

if __name__ == "__main__":
    from maze_gen_prims import PrimsMaze

    # Map sizes from 2x2 to 50x50
    size_min, size_max = 2, 50
    runs_per_size = 1000
    start = (0, 0)

    sizes = list(range(size_min, size_max+1))
    avg_times, avg_nodes, avg_memory, avg_path = [], [], [], []

    print(f"Running {runs_per_size} A* trials per size from {size_min} to {size_max}...")
    for idx, size in enumerate(sizes, 1):
        t_list, n_list, m_list, p_list = [], [], [], []
        maze = PrimsMaze(size).create_maze(start)
        graph = mat2graph(maze)
        free_cells = list(graph.keys())
        for _ in range(runs_per_size):
            goal = random.choice(free_cells)
            t, ne, ms, pl = A_star_stats(graph, start, goal)
            t_list.append(t)
            n_list.append(ne)
            m_list.append(ms)
            p_list.append(pl)
        avg_times.append(np.mean(t_list))
        avg_nodes.append(np.mean(n_list))
        avg_memory.append(np.mean(m_list))
        avg_path.append(np.mean(p_list))
        print(f"Size {size} [{idx}/{len(sizes)}] done", end='\r', flush=True)
    print()  # newline

    # Create DataFrame for table representation
    df = pd.DataFrame({
        'Maze Size': sizes,
        'Avg Time (s)': avg_times,
        'Avg Nodes': avg_nodes,
        'Avg Memory': avg_memory,
        'Avg Path Length': avg_path
    })

    # Display table
    print("\n===== A* Simulation Results =====")
    print(df.to_string(index=False))

    # Plot averages vs size
    plt.figure(figsize=(12, 7))
    plt.plot(sizes, avg_nodes, marker='s', label='Avg Nodes Explored')
    plt.plot(sizes, avg_memory, marker='^', label='Avg Memory Footprint')
    plt.plot(sizes, avg_path, marker='d', label='Avg Path Length')
    plt.title(f"A* Avg Nodes/Memory/Path vs Maze Size ({runs_per_size} trials)")
    plt.xlabel('Maze Size (N x N)')
    plt.ylabel('Average Metric')
    plt.xticks(np.arange(size_min, size_max+1, 4))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Separate plot for Avg Time
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, avg_times, marker='o', color='tab:orange', label='Avg Execution Time (s)')
    plt.title(f"A* Avg Execution Time vs Maze Size ({runs_per_size} trials)")
    plt.xlabel('Maze Size (N x N)')
    plt.ylabel('Average Time (s)')
    plt.xticks(np.arange(size_min, size_max+1, 4))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        print("Plot display interrupted. Closing.")
        plt.close('all')
