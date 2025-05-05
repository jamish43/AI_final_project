import numpy as np
from collections import defaultdict
import heapq
import time
import random
import pandas as pd
import matplotlib.pyplot as plt

# === 3D Environment Utilities ===

def mat3graph(grid):
    """
    Convert a 3D boolean occupancy grid into an adjacency list.
    grid: numpy array of shape (X,Y,Z), True=free, False=obstacle
    returns: dict mapping (x,y,z) to list of neighbor tuples
    """
    X, Y, Z = grid.shape
    graph = defaultdict(list)
    # 6-connected neighbors
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if not grid[x,y,z]:
                    continue
                for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    nx, ny, nz = x+dx, y+dy, z+dz
                    if 0 <= nx < X and 0 <= ny < Y and 0 <= nz < Z and grid[nx,ny,nz]:
                        graph[(x,y,z)].append((nx,ny,nz))
    return graph


def heuristic_euclidean(node, goal):
    return np.linalg.norm(np.array(node) - np.array(goal))

def heuristic_manhattan(node, goal):
    return sum(abs(a-b) for a,b in zip(node, goal))


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# === Standard A* in 3D ===

def A_star_stats_3d(graph, start, goal):
    t0 = time.time()
    open_set = {start}
    closed_set = set()
    came_from = {}
    gscore = {node: float('inf') for node in graph}
    gscore[start] = 0.0
    fscore = {node: float('inf') for node in graph}
    fscore[start] = heuristic_euclidean(start, goal)
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
        for nbr in graph[current]:
            if nbr in closed_set:
                continue
            tentative = gscore[current] + 1
            if tentative < gscore.get(nbr, float('inf')):
                came_from[nbr] = current
                gscore[nbr] = tentative
                fscore[nbr] = tentative + heuristic_euclidean(nbr, goal)
                open_set.add(nbr)
    elapsed = time.time() - t0
    path_len = len(reconstruct_path(came_from, current))
    return elapsed, nodes_explored, max_space, path_len


def run_astar_3d(size_min=2, size_max=10, trials=100):
    print("\n=== 3D A* Simulations ===")
    sizes = list(range(size_min, size_max+1))
    results = []
    for size in sizes:
        grid = np.ones((size,size,size), dtype=bool)
        graph = mat3graph(grid)
        free = list(graph.keys())
        stats = {'Size': size, 'Time': [], 'Nodes': [], 'Memory': [], 'PathLen': []}
        for _ in range(trials):
            start = (0,0,0)
            goal = random.choice(free)
            t, n, m, p = A_star_stats_3d(graph, start, goal)
            stats['Time'].append(t)
            stats['Nodes'].append(n)
            stats['Memory'].append(m)
            stats['PathLen'].append(p)
        results.append({
            'Size': size,
            'Avg Time (s)': np.mean(stats['Time']),
            'Avg Nodes': np.mean(stats['Nodes']),
            'Avg Memory': np.mean(stats['Memory']),
            'Avg PathLen': np.mean(stats['PathLen'])
        })
        print(f"A* 3D size={size}: time={results[-1]['Avg Time (s)']:.6f}s", end='\r')
    df = pd.DataFrame(results)
    print("\n3D A* Results:")
    print(df.to_string(index=False))
    return df

# === Memory-Bounded A* in 3D ===

def MA_star_stats_3d(graph, start, goal, memory_limit):
    t0 = time.time()
    open_set = {start}
    closed_set = set()
    came_from = {}
    gscore = defaultdict(lambda: float('inf'))
    fscore = defaultdict(lambda: float('inf'))
    gscore[start] = 0; fscore[start] = heuristic_euclidean(start, goal)
    nodes_explored = 0; max_mem = 0
    while open_set:
        nodes_explored += 1
        current = min(open_set, key=lambda n: fscore[n])
        if current == goal: break
        open_set.remove(current); closed_set.add(current)
        max_mem = max(max_mem, len(open_set)+len(closed_set))
        for nbr in graph[current]:
            if nbr in closed_set: continue
            tg = gscore[current] + 1
            if tg < gscore[nbr]:
                came_from[nbr] = current
                gscore[nbr] = tg
                fscore[nbr] = tg + heuristic_euclidean(nbr, goal)
                open_set.add(nbr)
                if len(open_set)+len(closed_set) > memory_limit:
                    worst = max(open_set, key=lambda n: fscore[n])
                    open_set.remove(worst)
    elapsed = time.time() - t0
    path_len = len(reconstruct_path(came_from, current))
    return elapsed, nodes_explored, max_mem, path_len


def run_mastar_3d(size_min=2, size_max=10, trials=100, memory_frac=0.5):
    print("\n=== 3D MA* Simulations ===")
    results = []
    for size in range(size_min, size_max+1):
        grid = np.ones((size,size,size), dtype=bool)
        graph = mat3graph(grid)
        limit = max(10, int(len(graph)*memory_frac))
        free = list(graph.keys())
        stats = {'Time':[], 'Nodes':[], 'Memory':[], 'PathLen':[]}
        for _ in range(trials):
            t,n,m,p = MA_star_stats_3d(graph,(0,0,0), random.choice(free), limit)
            stats['Time'].append(t); stats['Nodes'].append(n)
            stats['Memory'].append(m); stats['PathLen'].append(p)
        results.append({
            'Size': size,
            'Avg Time (s)': np.mean(stats['Time']),
            'Avg Nodes': np.mean(stats['Nodes']),
            'Avg Memory': np.mean(stats['Memory']),
            'Avg PathLen': np.mean(stats['PathLen'])
        })
        print(f"MA* 3D size={size}: time={results[-1]['Avg Time (s)']:.6f}s", end='\r')
    df = pd.DataFrame(results)
    print("\n3D MA* Results:")
    print(df.to_string(index=False))
    return df

# === Simplified Memory-Bounded A* (SMA*) in 3D ===

def sma_star_stats_3d(graph, start, goal, memory_limit):
    """
    Simplified MA*: prune worst frontier node without backups for 3D.
    """
    t0 = time.time()
    open_set = {start}
    closed_set = set()
    came_from = {}
    gscore = defaultdict(lambda: float('inf'))
    fscore = defaultdict(lambda: float('inf'))
    gscore[start] = 0; fscore[start] = heuristic_euclidean(start, goal)
    nodes_explored = 0; max_mem = 0
    while open_set:
        nodes_explored += 1
        # select lowest f
        current = min(open_set, key=lambda n: fscore[n])
        if current == goal:
            break
        open_set.remove(current)
        closed_set.add(current)
        max_mem = max(max_mem, len(open_set) + len(closed_set))
        for nbr in graph[current]:
            if nbr in closed_set:
                continue
            tg = gscore[current] + 1
            if tg < gscore[nbr]:
                came_from[nbr] = current
                gscore[nbr] = tg
                fscore[nbr] = tg + heuristic_euclidean(nbr, goal)
                open_set.add(nbr)
                # simplified prune: drop worst if exceeding memory
                if len(open_set) + len(closed_set) > memory_limit:
                    worst = max(open_set, key=lambda n: fscore[n])
                    open_set.remove(worst)
    elapsed = time.time() - t0
    path_len = len(reconstruct_path(came_from, current))
    return elapsed, nodes_explored, max_mem, path_len


def run_smastar_3d(size_min=2, size_max=10, trials=100, memory_frac=0.5):
    print("=== 3D SMA* Simulations ===")
    results = []
    for size in range(size_min, size_max+1):
        grid = np.ones((size,size,size), dtype=bool)
        graph = mat3graph(grid)
        limit = max(10, int(len(graph) * memory_frac))
        free = list(graph.keys())
        stats = {'Time': [], 'Nodes': [], 'Memory': [], 'PathLen': []}
        for _ in range(trials):
            t, n, m, p = sma_star_stats_3d(graph, (0,0,0), random.choice(free), limit)
            stats['Time'].append(t)
            stats['Nodes'].append(n)
            stats['Memory'].append(m)
            stats['PathLen'].append(p)
        results.append({
            'Size': size,
            'Avg Time (s)': np.mean(stats['Time']),
            'Avg Nodes': np.mean(stats['Nodes']),
            'Avg Memory': np.mean(stats['Memory']),
            'Avg PathLen': np.mean(stats['PathLen'])
        })
        print(f"SMA* 3D size={size}: time={results[-1]['Avg Time (s)']:.6f}s", end='')
    df = pd.DataFrame(results)
    print("3D SMA* Results:")
    print(df.to_string(index=False))
    return df

# === IDA* in 3D ===

def IDA_star_stats_3d(size, start, goal):
    dirs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    nodes_explored = 0; max_depth = 0
    def h(node): return heuristic_manhattan(node, goal)
    def dfs(node, g, bound, visited):
        nonlocal nodes_explored, max_depth
        nodes_explored += 1; max_depth = max(max_depth, g)
        f = g + h(node)
        if f > bound: return False, f
        if node == goal: return True, g
        min_b = float('inf')
        for dx,dy,dz in dirs:
            nbr = (node[0]+dx, node[1]+dy, node[2]+dz)
            if 0<=nbr[0]<size and 0<=nbr[1]<size and 0<=nbr[2]<size and nbr not in visited:
                visited.add(nbr)
                found,val = dfs(nbr, g+1, bound, visited)
                if found: return True, val
                min_b = min(min_b, val)
                visited.remove(nbr)
        return False, min_b
    bound = h(start); path_len=0; t0=time.time()
    while True:
        found, val = dfs(start, 0, bound, {start})
        if found: path_len = val; break
        if val == float('inf'): break
        bound = val
    return time.time()-t0, nodes_explored, max_depth, path_len


def run_idastar_3d(size_min=2, size_max=10, trials=100):
    print("\n=== 3D IDA* Simulations ===")
    results=[]
    for size in range(size_min, size_max+1):
        stats={'Time':[], 'Nodes':[], 'Depth':[], 'PathLen':[]}
        for _ in range(trials):
            start=(0,0,0);
            goal=(random.randrange(size),random.randrange(size),random.randrange(size))
            t,n,d,p = IDA_star_stats_3d(size, start, goal)
            stats['Time'].append(t); stats['Nodes'].append(n)
            stats['Depth'].append(d); stats['PathLen'].append(p)
        results.append({
            'Size':size,
            'Avg Time (s)':np.mean(stats['Time']),
            'Avg Nodes':np.mean(stats['Nodes']),
            'Avg Depth':np.mean(stats['Depth']),
            'Avg PathLen':np.mean(stats['PathLen'])
        })
        print(f"IDA* 3D size={size}: time={results[-1]['Avg Time (s)']:.6f}s", end='\r')
    df=pd.DataFrame(results)
    print("\n3D IDA* Results:")
    print(df.to_string(index=False))
    return df

# === Weighted A* in 3D ===

def weighted_astar_stats_3d(size, start, goal, weight):
    def h(node): return heuristic_manhattan(node, goal)
    open_heap=[(weight*h(start), start)]
    gscore={start:0}
    fscore={start:weight*h(start)}
    open_set={start}; closed_set=set()
    nodes_explored=0; max_mem=0; t0=time.time()
    while open_heap:
        _,current=heapq.heappop(open_heap)
        if current in closed_set: continue
        open_set.discard(current); nodes_explored+=1
        if current==goal:
            return time.time()-t0, nodes_explored, max_mem, gscore[current]
        closed_set.add(current)
        max_mem=max(max_mem,len(open_set)+len(closed_set))
        x,y,z=current
        for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nbr=(x+dx,y+dy,z+dz)
            if not all(0<=c<size for c in nbr): continue
            tentative=gscore[current]+1
            if tentative<gscore.get(nbr,float('inf')):
                gscore[nbr]=tentative
                fs=tentative + weight*h(nbr)
                fscore[nbr]=fs
                heapq.heappush(open_heap,(fs,nbr)); open_set.add(nbr)
    return time.time()-t0,nodes_explored,max_mem,None


def run_weighted_3d(size_min=2, size_max=10, trials=100, weights=None):
    print("\n=== 3D Weighted A* Simulations ===")
    if weights is None: weights=[1.0,1.5,2.0,2.5]
    for w in weights:
        results=[]
        for size in range(size_min, size_max+1):
            stats={'Time':[], 'Nodes':[], 'Memory':[], 'PathLen':[]}
            for _ in range(trials):
                start=(0,0,0)
                goal=(random.randrange(size),random.randrange(size),random.randrange(size))
                t,n,m,p = weighted_astar_stats_3d(size, start, goal, w)
                stats['Time'].append(t); stats['Nodes'].append(n)
                stats['Memory'].append(m); stats['PathLen'].append(p if p else 0)
            results.append({
                'Size':size,
                'Avg Time (s)':np.mean(stats['Time']),
                'Avg Nodes':np.mean(stats['Nodes']),
                'Avg Memory':np.mean(stats['Memory']),
                'Avg PathLen':np.mean(stats['PathLen'])
            })
            print(f"wA*3D w={w} size={size}: time={results[-1]['Avg Time (s)']:.6f}s", end='\r')
        df=pd.DataFrame(results)
        print(f"\n3D Weighted A* w={w} Results:")
        print(df.to_string(index=False))
    return


if __name__ == '__main__':
    run_astar_3d(size_min=2, size_max=25, trials=100)
    run_mastar_3d(size_min=2, size_max=25, trials=100)
    run_smastar_3d(size_min=2, size_max=25, trials=100)
    run_idastar_3d(size_min=2, size_max=25, trials=100)
    run_weighted_3d(size_min=2, size_max=25, trials=100)

