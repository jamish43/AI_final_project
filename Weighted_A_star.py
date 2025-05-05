import numpy as np
import matplotlib.pyplot as plt
import heapq
import time
import random
import pandas as pd

# === Weighted A* Implementation ===
def weighted_astar_stats(size, start, goal, weight):
    # Helper functions
    def offset(x, y):
        return y * size + x
    def coords(off):
        return (off % size, off // size)
    def h(x, y):
        return abs(x - goal[0]) + abs(y - goal[1])

    start_off = offset(*start)
    goal_off = offset(*goal)

    gscore = {start_off: 0}
    fscore = {start_off: weight * h(*start)}

    open_heap = [(fscore[start_off], start_off)]
    open_set = {start_off}
    closed_set = set()

    nodes_explored = 0
    max_mem = 0
    t0 = time.time()

    while open_heap:
        f, current = heapq.heappop(open_heap)
        if current in closed_set:
            continue
        open_set.discard(current)
        nodes_explored += 1

        if current == goal_off:
            length = gscore[current]
            elapsed = time.time() - t0
            return elapsed, nodes_explored, max_mem, length

        closed_set.add(current)
        max_mem = max(max_mem, len(open_set) + len(closed_set))

        cx, cy = coords(current)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < size and 0 <= ny < size):
                continue
            neighbor = offset(nx, ny)
            if neighbor in closed_set:
                continue
            tentative_g = gscore[current] + 1
            if tentative_g < gscore.get(neighbor, float('inf')):
                gscore[neighbor] = tentative_g
                fscore_neighbor = tentative_g + weight * h(nx, ny)
                fscore[neighbor] = fscore_neighbor
                heapq.heappush(open_heap, (fscore_neighbor, neighbor))
                open_set.add(neighbor)

    elapsed = time.time() - t0
    return elapsed, nodes_explored, max_mem, None

# === Simulation Loop over weights with Table ===
def run_weighted_simulations(size_min=2, size_max=50, trials=1000, weights=None):
    if weights is None:
        weights = [1.0, 1.5, 2.0, 2.5]
    sizes = list(range(size_min, size_max + 1))

    for w in weights:
        avg_times, avg_nodes, avg_memory, avg_paths = [], [], [], []
        print(f"Running Weighted A* w={w} ({trials} trials) sizes {size_min}-{size_max}...")
        for size in sizes:
            coords = [(x, y) for y in range(size) for x in range(size)]
            times, nodes, mems, paths = [], [], [], []
            for _ in range(trials):
                goal = random.choice(coords)
                t, ne, mu, pl = weighted_astar_stats(size, (0, 0), goal, w)
                times.append(t)
                nodes.append(ne)
                mems.append(mu)
                paths.append(pl if pl is not None else 0)
            avg_times.append(np.mean(times))
            avg_nodes.append(np.mean(nodes))
            avg_memory.append(np.mean(mems))
            avg_paths.append(np.mean(paths))

        # Create and print table
        df = pd.DataFrame({
            'Maze Size (N x N)':    sizes,
            'Avg Time (s)':          avg_times,
            'Avg Nodes Explored':    avg_nodes,
            'Avg Memory Footprint':  avg_memory,
            'Avg Path Length':       avg_paths
        })
        print(f"\n===== Weighted A* (w={w}) Simulation Results =====")
        print(df.to_string(index=False))

        # Plot metrics vs size
        plt.figure(figsize=(12, 6))
        plt.plot(sizes, avg_nodes, marker='o', label='Avg Nodes')
        plt.plot(sizes, avg_memory, marker='s', label='Avg Memory')
        plt.plot(sizes, avg_paths, marker='d', label='Avg Path Length')
        plt.title(f"Weighted A* Metrics (w={w}) vs Maze Size")
        plt.xlabel('Maze Size N x N')
        plt.ylabel('Average Metric')
        plt.xticks(np.arange(size_min, size_max + 1, 4))
        plt.legend(); plt.grid(True); plt.tight_layout()

        # Plot execution time separately
        plt.figure(figsize=(10, 5))
        plt.plot(sizes, avg_times, marker='^', color='tab:orange', label='Avg Time (s)')
        plt.title(f"Weighted A* Execution Time (w={w}) vs Maze Size")
        plt.xlabel('Maze Size N x N')
        plt.ylabel('Time (s)')
        plt.xticks(np.arange(size_min, size_max + 1, 4))
        plt.legend(); plt.grid(True); plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    run_weighted_simulations(size_min=2, size_max=50, trials=1000, weights=[1.0, 1.5, 2.0, 2.5])
#edited 