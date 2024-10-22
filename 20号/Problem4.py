import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
from matplotlib.patches import Circle, RegularPolygon
import itertools
import pandas as pd
import os
def parse_map(map_data):
    lines = map_data.strip().split("\n")
    grid = []
    for line in lines:
        grid.append(line.split())
    return np.array(grid)
def parse_tasks(tasks_data):
    lines = tasks_data.strip().split("\n")
    tasks = []
    tot = 0
    for line in lines:
        start_x, start_y, end_x, end_y = map(int, line.split())
        tot = tot + 1
        tasks.append(((start_x, start_y), (end_x, end_y)))
    return tasks
def parse_agents(agents_data):
    lines = agents_data.strip().split("\n")
    agent = []
    tot = 0
    for line in lines:
        x, y = map(int, line.split())
        tot = tot + 1
        agent.append((x, y))
    return agent

map_data = """
. . . . . . . . 
. @ . . @ . @ . 
. . . . . . . . 
. . . . . . . . 
@ . . . . . . . 
. . . . . . . . 
. . . . @ . . . 
. @ . . . . . . 
"""
tasks_data = """
1 7 3 3
5 0 7 5
7 4 5 2
4 5 6 7
6 5 0 6
5 2 2 4
0 4 7 2
4 1 1 2
"""
agents_data = """·
0 0
7 0
"""
grid = parse_map(map_data)
tasks = parse_tasks(tasks_data)
agents = parse_agents(agents_data)

must =[[0,1,2],[]]
#must =[[0,1,5],[]]
#must =[[],[0,6,7,10,17,23,32,40],[],[1,5,8,20],[]]

a = [1, 0]
#a = [0, 1]
#a = [0, 3, 2, 4, 1]

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
def time_a_star_search(grid, start, goal, other_paths, tt = 0):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]
    close_set = set()
    came_from = {}
    gscore = {(start[0], start[1], tt): 0}
    fscore = {(start[0], start[1], tt): heuristic(start, goal)}
    open_set = PriorityQueue()
    open_set.put((fscore[(start[0], start[1], tt)], (start[0], start[1], tt)))
    while not open_set.empty():
        _, current = open_set.get()
        x, y, t = current

        if (x, y) == goal:
            path = []
            while current in came_from:
                path.append((current[0], current[1]))
                current = came_from[current]
            path.append((start[0], start[1]))
            return path[::-1], t  # 返回路径和时间消耗

        close_set.add(current)
        for i, j in neighbors:
            neighbor = x + i, y + j, t + 1
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0]][neighbor[1]] == '@': continue

                conflict = False
                for path in other_paths:
                    """
                    if(len(path) == 0):break
                    if (neighbor[0], neighbor[1]) == goal:
                        for i in range(0, len(path)):
                            if(path[i] == goal and i >= t + 1):
                                conflict = True
                                break
                    if conflict:
                        break
                    """
                    if len(path) > t + 1 and path[t + 1] == (neighbor[0], neighbor[1]):
                        conflict = True
                        break
                    """
                    if len(path) <= t + 1 and path[-1] == (neighbor[0], neighbor[1]):
                        conflict = True
                        break
                    """
                    if len(path) > t and len(path) > t + 1 and path[t] == (neighbor[0], neighbor[1]) and path[t + 1] == (x, y):
                        conflict = True
                        break
                if conflict: continue

                tentative_g_score = gscore[current] + 1

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0): continue

                if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in open_set.queue]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic((neighbor[0], neighbor[1]), goal)
                    open_set.put((fscore[neighbor], neighbor))

    return False, float('inf')

def check(x):
    paths = [[] for _ in range(len(agents))]
    paths2 = [[] for _ in range(len(agents))]
    vis = [0 for _ in range(len(tasks))]
    cnt = 0
    for ii in range(0, len(a)):
        i1 = a[ii]
        agent = agents[i1]
        if cnt == len(tasks): break
        start1 = agent
        start2 = -1
        goal2 = -1
        id = -1
        minw = 100000
        sumt = 0
        other_paths = []
        if ii != 0:
            for i in range(0, ii - 1):
                other_paths.append(paths[a[i]])

        for i2, task in enumerate(tasks):
            start, goal = task
            if vis[i2]: continue

            flag = False
            for i3 in range(0, len(a)):
                if flag: break
                if i3 == i1: continue
                for i4 in must[i3]:
                    if i4 == i2:
                        flag = True
                        break
            if flag: continue
            path, t = time_a_star_search(grid, start1, start, other_paths, sumt)
            if minw > t:
                id = i2
                start2 = start
                goal2 = goal
                minw = t
        if goal2 == -1: continue

        path1, t1 = time_a_star_search(grid, start1, start2, other_paths, sumt)
        path2, t2 = time_a_star_search(grid, start2, goal2, other_paths, t1)

        if t2 > x: continue

        for p in path1: paths[i1].append((p))
        for p in path2[1:]: paths[i1].append((p))
        paths2[i1].append(path1)
        paths2[i1].append(path2)
        vis[id] = 1
        cnt = cnt + 1
        sumt = t2
        start1 = goal2
        while cnt != len(tasks) and sumt <= x:
            start2 = -1
            goal2 = -1
            id = -1
            minw = 100000

            for i2, task in enumerate(tasks):
                start, goal = task
                if (vis[i2]): continue

                flag = False
                for i3 in range(0, len(a)):
                    if flag: break
                    if i3 == i1: continue
                    for i4 in must[i3]:
                        if i4 == i2:
                            flag = True
                            break
                if flag: continue
                path, t = time_a_star_search(grid, start1, start, other_paths, sumt)
                if minw > t:
                    id = i2
                    start2 = start
                    goal2 = goal
                    minw = t
            if goal2 == -1: break

            path1, t1 = time_a_star_search(grid, start1, start2, other_paths, sumt)
            path2, t2 = time_a_star_search(grid, start2, goal2, other_paths, t1)
            if t2 > x: break
            for p in path1[1:]: paths[i1].append((p))
            for p in path2[1:]: paths[i1].append((p))
            paths2[i1].append(path1)
            paths2[i1].append(path2)
            vis[id] = 1
            cnt = cnt + 1
            sumt = t2
            start1 = goal2

    flag = False
    if cnt == len(tasks): flag = True

    return flag, paths, paths2
def solve():
    l = 0
    r = 1250
    while(l < r):
        mid = (l + r) // 2
        print(mid)
        flag, paths, paths2 = check(mid)
        if flag: r = mid
        else: l = mid + 1
    flag, paths, paths2 = check(l)

    print(l)
    for i, p in enumerate(paths): print(len(p) - 1, i + 1, p)
    for p in paths2:
        print(len(p))
        for pp in p: print(pp)

    file_path = "results_第四问 8.xlsx"
    if os.path.exists(file_path):
        with open(file_path, 'w') as file:
            pass

    data = {
        "机器人编号": [],
        "位置列表": [],
        "任务列表": [],
        "时间开销": []
    }

    for i, p in enumerate(paths, 1):
        data["机器人编号"].append(i)
        data["位置列表"].append(p)
        data["时间开销"].append(len(p)-1)
    for p in paths2:
        data["任务列表"].append(p)

    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)

solve()
