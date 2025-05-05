#!/usr/bin/env python
import os
import sys
import time
import random
import rospy
import heapq
import numpy as np
from collections import defaultdict
from geometry_msgs.msg import Twist, Vector3

# === 3D Environment Utilities ===

def mat3graph(grid):
    """
    Convert a 3D boolean occupancy grid into an adjacency list.
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

# === 3D Path-Finding Algorithms ===

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

sma_star_path_3d = MA_star_path_3d  # simplified pruning

def IDA_star_path_3d(size, start, goal):
    dirs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    def h(n): return heuristic_manhattan(n, goal)
    def dfs(node, g, bound, visited):
        f = g + h(node)
        if f > bound: return None, f
        if node == goal: return [node], g
        min_b = float('inf')
        for dx, dy, dz in dirs:
            nbr = (node[0]+dx, node[1]+dy, node[2]+dz)
            if any(c<0 or c>=size for c in nbr) or nbr in visited: continue
            visited.add(nbr)
            path, val = dfs(nbr, g+1, bound, visited)
            visited.remove(nbr)
            if path is not None: return [node] + path, val
            min_b = min(min_b, val)
        return None, min_b

    bound = h(start)
    while True:
        path, val = dfs(start, 0, bound, {start})
        if path is not None: return path
        if val == float('inf'): return []
        bound = val

def weighted_astar_path_3d(size, start, goal, weight):
    open_heap = [(weight * heuristic_manhattan(start, goal), start)]
    came_from = {}
    gscore = {start: 0}
    closed = set()
    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed: continue
        if current == goal:
            return reconstruct_path(came_from, current)
        closed.add(current)
        x, y, z = current
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nbr = (x+dx, y+dy, z+dz)
            if not all(0<=c<size for c in nbr): continue
            tg = gscore[current] + 1
            if tg < gscore.get(nbr, float('inf')):
                came_from[nbr] = current
                gscore[nbr] = tg
                f = tg + weight * heuristic_manhattan(nbr, goal)
                heapq.heappush(open_heap, (f, nbr))
    return []

# === Command Generation ===

def generate_3d_ros_commands(path):
    """
    Convert a 3D coordinate path into ROS teleop commands.
    Returns list of (cmd_name, value) tuples.
    """
    commands = []
    # horizontal orientations: +X, +Y, -X, -Y
    dirs = [(1,0), (0,1), (-1,0), (0,-1)]
    ori = 0  # start facing +X

    for (x0,y0,z0), (x1,y1,z1) in zip(path, path[1:]):
        dx, dy, dz = x1-x0, y1-y0, z1-z0
        if dz == 1:
            commands.append(("up", None))
        elif dz == -1:
            commands.append(("down", None))
        else:
            target = dirs.index((dx, dy))
            turn = (target - ori) % 4
            if turn == 1:
                commands.append(("turn_right", None))
            elif turn == 2:
                commands.extend([("turn_right", None), ("turn_right", None)])
            elif turn == 3:
                commands.append(("turn_left", None))
            ori = target
            commands.append(("move_forward", None))
    return commands

# === ROS Teleop Executor ===

class KeyBoardVehicleTeleop:
    def __init__(self, commands):
        self.settings = termios.tcgetattr(sys.stdin)
        self.speed = 1
        self.l = Vector3(0,0,0)
        self.a = Vector3(0,0,0)
        self.linear_increment = 0.05
        self.linear_limit = 1
        self.angular_increment = 0.05
        self.angular_limit = 0.5
        self.commands = commands
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        rospy.loginfo("Executing %d commands..."%len(commands))
        self._execute_commands()

    def _speed_windup(self, speed, inc, limit, reverse):
        if reverse:
            speed -= inc * self.speed
            return max(speed, -limit*self.speed)
        else:
            speed += inc * self.speed
            return min(speed, limit*self.speed)

    def _execute_repeated(self, fn, reps, hz):
        rate = rospy.Rate(hz)
        for _ in range(reps):
            fn()
            rate.sleep()

    def _execute_command(self, cmd):
        twist = Twist()
        if cmd=="move_forward":
            rospy.loginfo("Forward")
            def fwd():
                self.l.x = self._speed_windup(self.l.x, self.linear_increment, self.linear_limit, False)
                twist.linear, twist.angular = self.l, self.a
                self.pub.publish(twist)
            self._execute_repeated(fwd, 10, 10)

        elif cmd=="turn_left":
            rospy.loginfo("Turn Left")
            def yl(): 
                self.a.z = self._speed_windup(self.a.z, self.angular_increment, self.angular_limit, False)
                twist.linear, twist.angular = self.l, self.a
                self.pub.publish(twist)
            def yr():
                self.a.z = self._speed_windup(self.a.z, self.angular_increment, self.angular_limit, True)
                twist.linear, twist.angular = self.l, self.a
                self.pub.publish(twist)
            self._execute_repeated(yl, 47, 10)
            self._execute_repeated(yr, 20, 10)

        elif cmd=="turn_right":
            rospy.loginfo("Turn Right")
            def yr(): 
                self.a.z = self._speed_windup(self.a.z, self.angular_increment, self.angular_limit, True)
                twist.linear, twist.angular = self.l, self.a
                self.pub.publish(twist)
            def yl():
                self.a.z = self._speed_windup(self.a.z, self.angular_increment, self.angular_limit, False)
                twist.linear, twist.angular = self.l, self.a
                self.pub.publish(twist)
            self._execute_repeated(yr, 47, 10)
            self._execute_repeated(yl, 20, 10)

        elif cmd=="up":
            rospy.loginfo("Ascend")
            def up_fn():
                self.l.z = self._speed_windup(self.l.z, self.linear_increment, self.linear_limit, False)
                twist.linear, twist.angular = self.l, self.a
                self.pub.publish(twist)
            self._execute_repeated(up_fn, 10, 5)

        elif cmd=="down":
            rospy.loginfo("Descend")
            def down_fn():
                self.l.z = self._speed_windup(self.l.z, self.linear_increment, self.linear_limit, True)
                twist.linear, twist.angular = self.l, self.a
                self.pub.publish(twist)
            self._execute_repeated(down_fn, 10, 5)

    def _execute_commands(self):
        for cmd, _ in self.commands:
            self._execute_command(cmd)
            rospy.sleep(0.1)  # small pause between commands

# === Main ===

if __name__ == '__main__':
    rospy.init_node('3d_path_executor')
    size = 10
    grid = np.ones((size, size, size), dtype=bool)
    graph = mat3graph(grid)
    start = (0,0,0)
    goal  = (random.randrange(size),
             random.randrange(size),
             random.randrange(size))
    rospy.loginfo("Planning 3D path to %s", str(goal))

    mem_limit = int(len(graph)*0.5)
    path = A_star_path_3d(graph, start, goal)

    commands = generate_3d_ros_commands(path)
    rospy.loginfo("Generated %d commands", len(commands))

    teleop = KeyBoardVehicleTeleop(commands)
    rospy.loginfo("Finished execution")
