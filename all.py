import numpy as np
import matplotlib.pyplot as plt
import heapq
import time
import random
import pandas as pd
from collections import defaultdict
from maze_gen_prims import PrimsMaze

# === Standard A* (batch simulation & stats) ===

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def heuristic(cell, goal):
    # Euclidean distance
    return np.hypot(goal[0] - cell[0], goal[1] - cell[1])


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
    path_len = len(reconstruct_path(came_from, current))
    return exec_time, nodes_explored, max_space, path_len


def run_astar_simulations(size_min=2, size_max=50, trials=1000):
    print("\n=== Running Standard A* Simulations ===")
    sizes = list(range(size_min, size_max+1))
    avg_times, avg_nodes, avg_memory, avg_path = [], [], [], []
    for size in sizes:
        t_list, n_list, m_list, p_list = [], [], [], []
        maze = PrimsMaze(size).create_maze((0,0))
        graph = mat2graph(maze)
        free_cells = list(graph.keys())
        for _ in range(trials):
            goal = random.choice(free_cells)
            t, ne, ms, pl = A_star_stats(graph, (0,0), goal)
            t_list.append(t); n_list.append(ne); m_list.append(ms); p_list.append(pl)
        avg_times.append(np.mean(t_list))
        avg_nodes.append(np.mean(n_list))
        avg_memory.append(np.mean(m_list))
        avg_path.append(np.mean(p_list))
        print(f"A* size {size}: time={avg_times[-1]:.6f}s", end='\r')
    print()
    df = pd.DataFrame({
        'Maze Size': sizes,
        'Avg Time (s)': avg_times,
        'Avg Nodes': avg_nodes,
        'Avg Memory': avg_memory,
        'Avg Path Length': avg_path
    })
    print(df.to_string(index=False))
    # plots
    plt.figure(figsize=(12,6))
    plt.plot(sizes, avg_nodes, label='Avg Nodes')
    plt.plot(sizes, avg_memory, label='Avg Memory')
    plt.plot(sizes, avg_path, label='Avg Path Length')
    plt.title('A* Metrics vs Maze Size')
    plt.xlabel('Size'); plt.ylabel('Average'); plt.legend(); plt.grid(True)
    plt.figure(figsize=(8,4))
    plt.plot(sizes, avg_times, label='Avg Time (s)')
    plt.title('A* Execution Time vs Size')
    plt.xlabel('Size'); plt.ylabel('Time (s)'); plt.legend(); plt.grid(True)
    plt.show()


# === Memory-Bounded A* (MA*) ===

def memory_bounded_A_star_stats(graph, start, goal, memory_limit):
    t0 = time.time()
    open_set = {start}
    closed_set = set()
    came_from = {}
    gscore = defaultdict(lambda: float('inf'))
    fscore = defaultdict(lambda: float('inf'))
    gscore[start] = 0; fscore[start] = heuristic(start, goal)
    nodes_explored = 0; max_mem = 0
    while open_set:
        nodes_explored += 1
        current = min(open_set, key=lambda n: fscore[n])
        if current == goal:
            break
        open_set.remove(current); closed_set.add(current)
        max_mem = max(max_mem, len(open_set)+len(closed_set))
        for neighbor in graph[current]:
            if neighbor in closed_set: continue
            tentative = gscore[current] + 1
            if tentative < gscore[neighbor]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative
                fscore[neighbor] = tentative + heuristic(neighbor, goal)
                open_set.add(neighbor)
                if len(open_set)+len(closed_set) > memory_limit:
                    worst = max(open_set, key=lambda n: fscore[n])
                    open_set.remove(worst)
    elapsed = time.time() - t0
    path_len = len(reconstruct_path(came_from, current))
    return elapsed, nodes_explored, max_mem, path_len


def run_mastar_simulations(size_min=2, size_max=50, trials=1000, memory_frac=0.5):
    print("\n=== Running Memory-Bounded A* Simulations ===")
    sizes = list(range(size_min, size_max+1))
    avg_times, avg_nodes, avg_memory, avg_path = [], [], [], []
    for size in sizes:
        maze = PrimsMaze(size).create_maze((0,0)); graph = mat2graph(maze)
        limit = max(10, int(len(graph)*memory_frac))
        free = list(graph.keys())
        ts, ns, ms, ps = [], [], [], []
        for _ in range(trials):
            goal = random.choice(free)
            t, ne, mu, pl = memory_bounded_A_star_stats(graph, (0,0), goal, limit)
            ts.append(t); ns.append(ne); ms.append(mu); ps.append(pl)
        avg_times.append(np.mean(ts)); avg_nodes.append(np.mean(ns))
        avg_memory.append(np.mean(ms)); avg_path.append(np.mean(ps))
        print(f"MA* size {size}: time={avg_times[-1]:.6f}s", end='\r')
    print()
    df = pd.DataFrame({
        'Maze Size': sizes,
        'Avg Time (s)': avg_times,
        'Avg Nodes': avg_nodes,
        'Avg Memory': avg_memory,
        'Avg Path Length': avg_path
    })
    print(df.to_string(index=False))
    plt.figure(figsize=(12,6))
    plt.plot(sizes, avg_nodes, label='Avg Nodes')
    plt.plot(sizes, avg_memory, label='Avg Memory')
    plt.plot(sizes, avg_path, label='Avg Path Length')
    plt.title('MA* Metrics vs Size'); plt.xlabel('Size'); plt.legend(); plt.grid(True)
    plt.figure(figsize=(8,4))
    plt.plot(sizes, avg_times, label='Avg Time')
    plt.title('MA* Time vs Size'); plt.xlabel('Size'); plt.ylabel('Time (s)'); plt.legend(); plt.grid(True)
    plt.show()


# === Simplified MA* (SMA*) ===

def run_smastar_simulations(size_min=2, size_max=50, trials=1000, memory_frac=0.5):
    print("\n=== Running Simplified MA* Simulations ===")
    sizes = list(range(size_min, size_max+1))
    avg_times, avg_nodes, avg_memory, avg_path = [], [], [], []
    for size in sizes:
        maze = PrimsMaze(size).create_maze((0,0)); graph=mat2graph(maze)
        limit = max(10, int(len(graph)*memory_frac))
        free=list(graph.keys()); ts, ns, ms, ps=[],[],[],[]
        for _ in range(trials):
            goal=random.choice(free)
            t, ne, mu, pl = memory_bounded_A_star_stats(graph,(0,0),goal,limit)
            ts.append(t); ns.append(ne); ms.append(mu); ps.append(pl)
        avg_times.append(np.mean(ts)); avg_nodes.append(np.mean(ns))
        avg_memory.append(np.mean(ms)); avg_path.append(np.mean(ps))
        print(f"SMA* size {size}: time={avg_times[-1]:.6f}s", end='\r')
    print()
    df = pd.DataFrame({
        'Maze Size': sizes,
        'Avg Time (s)': avg_times,
        'Avg Nodes': avg_nodes,
        'Avg Memory': avg_memory,
        'Avg Path Length': avg_path
    })
    print(df.to_string(index=False))
    plt.figure(figsize=(12,6))
    plt.plot(sizes, avg_nodes, label='Avg Nodes')
    plt.plot(sizes, avg_memory, label='Avg Memory')
    plt.plot(sizes, avg_path, label='Avg Path Length')
    plt.title('SMA* Metrics vs Size'); plt.xlabel('Size'); plt.legend(); plt.grid(True)
    plt.figure(figsize=(8,4))
    plt.plot(sizes, avg_times, label='Avg Time')
    plt.title('SMA* Time vs Size'); plt.xlabel('Size'); plt.ylabel('Time (s)'); plt.legend(); plt.grid(True)
    plt.show()


# === Iterative Deepening A* (IDA*) ===

def ida_star_stats(size, start, goal):
    directions=[(-1,0),(1,0),(0,-1),(0,1)]
    nodes_explored=0; max_depth=0
    def h(node): return abs(node[0]-goal[0])+abs(node[1]-goal[1])
    def dfs(node,g,bound,visited):
        nonlocal nodes_explored,max_depth
        nodes_explored+=1; max_depth=max(max_depth,g)
        f=g+h(node)
        if f>bound: return False,f
        if node==goal: return True,g
        min_b=float('inf')
        for dx,dy in directions:
            nbr=(node[0]+dx,node[1]+dy)
            if 0<=nbr[0]<size and 0<=nbr[1]<size and nbr not in visited:
                visited.add(nbr)
                found,val=dfs(nbr,g+1,bound,visited)
                if found: return True,val
                min_b=min(min_b,val)
                visited.remove(nbr)
        return False,min_b
    
    bound=h(start); path_len=0; t0=time.time()
    while True:
        visited={start}
        found,val=dfs(start,0,bound,visited)
        if found: path_len=val; break
        if val==float('inf'): break
        bound=val
    return time.time()-t0,nodes_explored,max_depth,path_len


def run_idastar_simulations(size_min=2, size_max=50, trials=1000):
    print("\n=== Running IDA* Simulations ===")
    sizes=list(range(size_min,size_max+1))
    avg_times,avg_nodes,avg_depths,avg_paths=[],[],[],[]
    for size in sizes:
        ts,ns,ds,ps=[],[],[],[]
        coords=[(x,y) for x in range(size) for y in range(size)]
        for _ in range(trials):
            goal=random.choice(coords)
            t,ne,md,pl=ida_star_stats(size,(0,0),goal)
            ts.append(t); ns.append(ne); ds.append(md); ps.append(pl)
        avg_times.append(np.mean(ts)); avg_nodes.append(np.mean(ns))
        avg_depths.append(np.mean(ds)); avg_paths.append(np.mean(ps))
        print(f"IDA* size {size}: time={avg_times[-1]:.6f}s", end='\r')
    print()
    df=pd.DataFrame({
        'Map Size':sizes,
        'Avg Time (s)':avg_times,
        'Avg Nodes':avg_nodes,
        'Avg Max Depth':avg_depths,
        'Avg Path Length':avg_paths
    })
    print(df.to_string(index=False))
    plt.figure(figsize=(12,6))
    plt.plot(sizes,avg_nodes,label='Avg Nodes')
    plt.plot(sizes,avg_depths,label='Avg Depth')
    plt.plot(sizes,avg_paths,label='Avg Path Length')
    plt.title('IDA* Metrics vs Size'); plt.legend(); plt.grid(True)
    plt.figure(figsize=(8,4))
    plt.plot(sizes,avg_times,label='Avg Time')
    plt.title('IDA* Time vs Size'); plt.xlabel('Size'); plt.ylabel('Time (s)'); plt.legend(); plt.grid(True)
    plt.show()


# === Weighted A* ===

def weighted_astar_stats(size, start, goal, weight):
    def offset(x,y): return y*size+x
    def coords(o): return (o%size,o//size)
    def h(x,y): return abs(x-goal[0])+abs(y-goal[1])
    s_off=offset(*start); g_off=offset(*goal)
    gscore={s_off:0}; fscore={s_off:weight*h(*start)}
    open_heap=[(fscore[s_off],s_off)]; open_set={s_off}; closed_set=set()
    nodes_explored=0; max_mem=0; t0=time.time()
    while open_heap:
        f,c=heapq.heappop(open_heap)
        if c in closed_set: continue
        open_set.discard(c); nodes_explored+=1
        if c==g_off: return time.time()-t0,nodes_explored,max_mem,gscore[c]
        closed_set.add(c); max_mem=max(max_mem,len(open_set)+len(closed_set))
        cx,cy=coords(c)
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx,ny=cx+dx,cy+dy
            if not (0<=nx<size and 0<=ny<size): continue
            n_off=offset(nx,ny)
            if n_off in closed_set: continue
            tg=gscore[c]+1
            if tg<gscore.get(n_off,float('inf')):
                gscore[n_off]=tg
                fs=tg+weight*h(nx,ny)
                fscore[n_off]=fs
                heapq.heappush(open_heap,(fs,n_off)); open_set.add(n_off)
    return time.time()-t0,nodes_explored,max_mem,None


def run_weighted_simulations(size_min=2,size_max=50,trials=1000,weights=None):
    print("\n=== Running Weighted A* Simulations ===")
    if weights is None: weights=[1.0,1.5,2.0,2.5]
    sizes=list(range(size_min,size_max+1))
    for w in weights:
        avg_times,avg_nodes,avg_memory,avg_paths=[],[],[],[]
        for size in sizes:
            coords=[(x,y) for y in range(size) for x in range(size)]
            ts,ns,ms,ps=[],[],[],[]
            for _ in range(trials):
                goal=random.choice(coords)
                t,ne,mu,pl=weighted_astar_stats(size,(0,0),goal,w)
                ts.append(t); ns.append(ne); ms.append(mu); ps.append(pl or 0)
            avg_times.append(np.mean(ts)); avg_nodes.append(np.mean(ns))
            avg_memory.append(np.mean(ms)); avg_paths.append(np.mean(ps))
        df=pd.DataFrame({
            'Maze Size':sizes,
            'Avg Time (s)':avg_times,
            'Avg Nodes':avg_nodes,
            'Avg Memory':avg_memory,
            'Avg Path Length':avg_paths
        })
        print(f"\n--- Weighted A* w={w} ---")
        print(df.to_string(index=False))
        plt.figure(figsize=(12,6))
        plt.plot(sizes,avg_nodes,label='Avg Nodes')
        plt.plot(sizes,avg_memory,label='Avg Memory')
        plt.plot(sizes,avg_paths,label='Avg Path Length')
        plt.title(f'Weighted A* w={w} Metrics')
        plt.legend(); plt.grid(True)
        plt.figure(figsize=(8,4))
        plt.plot(sizes,avg_times,label='Avg Time')
        plt.title(f'Weighted A* w={w} Time')
        plt.xlabel('Size'); plt.ylabel('Time (s)'); plt.legend(); plt.grid(True)
    plt.show()


if __name__ == '__main__':
    run_astar_simulations()
    run_mastar_simulations()
    run_smastar_simulations()
    run_idastar_simulations()
    run_weighted_simulations()
