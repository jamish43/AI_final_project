import numpy as np
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
#import ace_tools as tools

# === Iterative Deepening A* (IDA*) ===

def ida_star_stats(size, start, goal):
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    nodes_explored = 0
    max_depth = 0

    def heuristic(node):
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    def dfs(node, g, bound, visited):
        nonlocal nodes_explored, max_depth
        nodes_explored += 1
        max_depth = max(max_depth, g)
        f = g + heuristic(node)
        if f > bound:
            return False, f
        if node == goal:
            return True, g
        min_bound = float('inf')
        for dx, dy in directions:
            neigh = (node[0] + dx, node[1] + dy)
            if 0 <= neigh[0] < size and 0 <= neigh[1] < size and neigh not in visited:
                visited.add(neigh)
                found, val = dfs(neigh, g+1, bound, visited)
                if found:
                    return True, val
                if val < min_bound:
                    min_bound = val
                visited.remove(neigh)
        return False, min_bound

    start_time = time.time()
    bound = heuristic(start)
    path_len = None
    while True:
        visited = {start}
        found, val = dfs(start, 0, bound, visited)
        if found:
            path_len = val
            break
        if val == float('inf'):
            path_len = 0
            break
        bound = val

    exec_time = time.time() - start_time
    return exec_time, nodes_explored, max_depth, path_len

# === Simulation and Table ===

def run_ida_star_simulations(size_min=2, size_max=50, trials=1000):
    sizes = list(range(size_min, size_max+1))
    avg_times, avg_nodes, avg_depths, avg_paths = [], [], [], []

    total = len(sizes)
    print(f"Running IDA* ({trials} trials) for sizes {size_min} to {size_max}...")
    for idx, size in enumerate(sizes, 1):
        t_list, n_list, d_list, p_list = [], [], [], []
        coords = [(x,y) for x in range(size) for y in range(size)]
        for _ in range(trials):
            goal = random.choice(coords)
            t, ne, md, pl = ida_star_stats(size, (0,0), goal)
            t_list.append(t)
            n_list.append(ne)
            d_list.append(md)
            p_list.append(pl)
        avg_times.append(np.mean(t_list))
        avg_nodes.append(np.mean(n_list))
        avg_depths.append(np.mean(d_list))
        avg_paths.append(np.mean(p_list))
        print(f"Size {size} [{idx}/{total}] avg_time={avg_times[-1]:.4f}", end='\r', flush=True)
    print()  # newline

    # Build DataFrame
    df = pd.DataFrame({
        'Map Size (N x N)': sizes,
        'Avg Time (s)': avg_times,
        'Avg Nodes Explored': avg_nodes,
        'Avg Max Depth': avg_depths,
        'Avg Path Length': avg_paths
    })

    # Display table
    print("\n===== IDA* Simulation Results =====")
    print(df.to_string(index=False))
    #tools.display_dataframe_to_user("IDA* Simulation Results", df)

    # Plot metrics
    plt.figure(figsize=(12, 6))
    plt.plot(sizes, avg_nodes, label='Avg Nodes Explored')
    plt.plot(sizes, avg_depths, label='Avg Max Depth')
    plt.plot(sizes, avg_paths, label='Avg Path Length')
    plt.title(f"IDA* Metrics vs Map Size ({trials} trials)")
    plt.xlabel('Map Size N x N')
    plt.ylabel('Average')
    plt.legend(); plt.grid(True); plt.tight_layout()

    plt.figure(figsize=(10, 5))
    plt.plot(sizes, avg_times, color='tab:orange', label='Avg Execution Time (s)')
    plt.title(f"IDA* Avg Execution Time vs Map Size ({trials} trials)")
    plt.xlabel('Map Size N x N')
    plt.ylabel('Time (s)')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_ida_star_simulations(size_min=2, size_max=50, trials=1000)
# editied for table