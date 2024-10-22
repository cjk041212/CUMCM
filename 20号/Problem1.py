import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
from matplotlib.patches import Circle, RegularPolygon
import pandas as pd
import os

# 解析地图
def parse_map(map_data):
    lines = map_data.strip().split("\n")
    grid = []
    for line in lines:
        grid.append(line.split())
    return np.array(grid)


# 解析任务
def parse_tasks(tasks_data):
    lines = tasks_data.strip().split("\n")[1:]  # 忽略任务数量行
    tasks = []
    for line in lines:
        start_x, start_y, end_x, end_y = map(int, line.split())
        tasks.append(((start_x, start_y), (end_x, end_y)))
    return tasks


# 启发式函数（曼哈顿距离）
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# 扩展A*算法以处理时间维度
def time_a_star_search(grid, start, goal, other_paths):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0),(0, 0)]
    close_set = set()
    came_from = {}
    gscore = {(start[0], start[1], 0): 0}
    fscore = {(start[0], start[1], 0): heuristic(start, goal)}
    open_set = PriorityQueue()
    open_set.put((fscore[(start[0], start[1], 0)], (start[0], start[1], 0)))

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
        for i, j in neighbors:  # 添加等待的可能性
            neighbor = x + i, y + j, t + 1
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0]][neighbor[1]] == '@':
                    continue

                # 检查是否与其他机器人路径冲突（点冲突）
                conflict = False
                for path in other_paths:
                    if len(path) > t + 1 and path[t + 1] == (neighbor[0], neighbor[1]):
                        conflict = True
                        break
                    # 检查边冲突
                    if len(path) > t and len(path) > t + 1 and path[t] == (neighbor[0], neighbor[1]) and path[t + 1] == (x, y):
                        conflict = True
                        break
                if conflict:
                    continue

                tentative_g_score = gscore[current] + 1

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue

                if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in open_set.queue]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic((neighbor[0], neighbor[1]), goal)
                    open_set.put((fscore[neighbor], neighbor))

    return False, float('inf')


# 处理数据
map_data = """
. . . . . . @ . . @ . . . @ @ @ . @ . . @ . . . . @ . . @ . . . . @ . . . . . . . . . . . . . . . @ . . . . . . . . . . @ @ . .
. . @ . . . . . @ @ . . @ . . . . . . . . . . . . . . . . . @ . . . . . . . . . @ . . . @ . . . @ . @ . @ . @ . . . . . . . . .
. . . @ . @ . . . . . . . . @ . . . . . . . . . . @ @ . . @ . . . . . @ @ . . . . . . @ . . . @ . . . @ . . . . . . . . . . . .
. . . . . . . . . . . . . . . @ @ . . . . @ . @ . . . @ . . . . . . . . . . . . @ . . . @ . . . . . @ . . . . . @ @ @ . . . . @
. . @ . . . . @ . . @ . . . . . . . . . . . . . . . . @ . . . . . . . @ @ @ . . . . . . @ . . . . . . . . @ . . . . . . . @ @ @
. . . . . @ @ . . @ . @ . . . . @ . . . . . . . . . . . . . @ . @ . . . @ . . . . . . @ . . . . . @ @ . @ . . . . . . . . . . .
. . @ . . . . . . @ . . . . @ . . @ . @ . @ . . . @ @ . . . @ . . . . @ . . . . . . . . . . . @ @ @ . . @ . . @ . . @ . . . . @
. . . @ @ @ . @ . . . . . @ . @ . . . . @ . . . @ @ . . . @ . . . . . . . . . . . . . . . @ . @ . . . . @ . . . . @ . . . . . @
@ @ . @ . . . @ . . . . . @ . . . . . @ . . . . . @ . . . . . . . . . . . . . @ @ @ . . @ . . . @ . . . . . . @ . . . . . . . .
. . . . . . . . . . @ . . . @ @ . . . . . . . . @ . . . @ . . . @ . . . . @ . . . . . . . . . @ . . . . . . . . . @ . . . . . .
. . . . @ . . . . . . . @ . . @ . . . . . . . . . . . . . . @ . . . . . . . . . . . . . . . . . . . . . . . . . @ . . . @ @ . .
. . . . . . . . @ . @ @ . . @ . . . . . . . . . . . . @ @ . @ @ @ . . . @ . . . . @ . . @ . . @ . @ @ . @ . . @ @ . @ . . @ . @
. @ . . @ @ @ . . . . . . . . . . . @ . . . . . . . . . . . . . . @ . . . @ . . . . . . . . . . @ @ @ @ . . . . . . @ . . @ . .
. @ . . . . @ . . . @ . . . . . @ @ . . . . . @ . . . . @ . . . @ . . . . @ . . . . . . . . . @ . . @ @ @ . . . . . @ @ . . . @
. . . . . . . @ . . . . . . . . . . . @ . . . @ . . . . . . . . . . . @ @ . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . @ @ . . . . @ . . . . @ . . @ . . @ . . @ @ . . . . . @ . @ . . . . . . . @ . . . . . . @ . . @ @ @ @ . . . . @ .
. . . . . . . @ . . . . . . . . . @ . @ . . @ . @ . . . . . . . . @ . . . . . . . . . . . . @ . . . . . . . . @ . . . @ . . . .
. . . @ . . . . . . . . . . . . . @ . . . . . . . . . . . . . @ . . . . @ . @ . . . . . . @ . . . @ . . . . . . . . . . . . . @
. . . . . . . . . . . . . . @ @ . . @ @ . . . . . . @ . . @ . . . . . . . . . . . . . . . . . . @ . . . . @ . . @ . . . . @ . .
@ . @ . . . . @ . . @ . . @ . @ . . @ . @ @ . . . . . . . . . . . . . . . . . . . . . @ @ . . . . . . . . . . . . . . . @ @ . .
. @ . . . . . . . @ . . . . . . . . . . @ . @ @ . . . . . . . . @ . . @ @ . . . . . @ . @ . . . . . . @ . . . . . @ . . . . . .
. . . @ . . @ . . . @ . . @ . . . @ . . @ . . . . . @ . . . . . . . . . . . . . . @ . @ . @ . . . . @ . . @ . . . . @ . . . . .
. . . . . . . . @ @ . . . . . . . . . . . . @ . . . . . @ . . . . . . . . @ @ . . . . . . . . . . . . . . . . . . . . . . . . .
. . @ . . . . @ . . . . . . . . . . . . . . @ . . . . . . . . . . . . . . . . . . . @ . . . . . @ . . . . . @ @ . . . . . . . .
. . . . @ . @ . . . . . . . @ . . @ . @ . . . . . . . . . . . @ . . . . . . . . . . . . . . . . . . . . . . . @ . . . . . @ @ .
. @ . . . . . @ @ . . . . . . . . . . . @ @ . @ . @ . @ . @ . @ @ . . . @ . . @ . @ . . . . . . . @ @ . . . . . . . . . @ . . .
. . . @ . . . @ . @ @ . . . . @ . . . . . . . . . @ @ . . . @ . . . . . @ . . . . . . . . . . . . . . . . . . . @ . . @ @ . . .
@ . @ @ . @ . . . . @ . . . . . . . @ . . . . . . . . . . . . . @ . . @ . . . . @ . . . . . . . . . . . . . . . @ . . . . . . .
. . . . . @ . . . . . . . . . . . . . . . . @ @ . . . . @ . @ . . . . . . . @ @ . . . . . . . . @ . . @ . . . . @ . . . . . @ .
. . @ @ . . . . . . @ . @ . @ . . . . . @ . . . @ @ . . @ . @ . . . . . @ . . . . . . . . . . . . . . . . . . . . . . @ . . . @
@ @ . . . . @ . . . @ . . . . . . . . . . . . . . . . @ . . . . . . . . . . . . . @ . . . . . . . @ . . . @ . . . @ . . @ . . @
. . . . . . . . . . . . . . . . . . . @ . @ . . . . . . . . . @ @ . . . . . @ . . . . . @ . . . . . @ . . . . @ . . . . . @ . .
@ . @ @ . . . . @ . @ . . @ . @ . . @ . . @ . . . . . . . . . . @ . . . . . . @ . @ . . @ . . @ . . . @ . . @ . . . . . . . . .
@ . . . . . @ . . . . . . . . . . . . . . . . @ . @ . @ . . . . . @ @ . @ . . . @ . . @ . . . . . @ . . @ . . . . . . . . . . .
. . . . . . . . . . . . . @ . . @ @ . . . @ . . @ . . . @ @ @ @ . . . . . @ . . @ . . . . . . . @ . . . @ . . . . @ @ . . . . @
@ . @ . . . . . . . @ . . . @ . . @ . . . . . . . . . . . @ . . @ . . . . . . . . . @ . . . . @ . . @ . . @ . @ . . . . . . . .
. @ @ @ . . @ . . @ . . . . . . . . . . . . . . . @ . . . . . . @ . . . . . . . . . . @ . . . @ . . . @ . @ . . @ . . @ @ . . @
. @ . . . . @ . . . . . . . . . . . . . . . @ . . . . . @ . . @ . . @ @ . . . . . . @ @ . . . . . . . . . . . . . . . @ @ . . .
@ . . . . . . @ . @ @ . @ . . @ @ . @ @ @ . . . @ . . . . . . . @ . . . . . . @ . . . . . . . . . @ . . . . @ . . . . @ . . . .
. . . . . . . . . @ @ @ . . . . . . . . @ @ . . . . . . . . . . . @ . . . . . @ . . . . @ . @ . @ . . . . @ . @ . . . @ . . . .
. . . . . . . . . . . . . . . . @ . @ . . @ @ . . . . . @ . . . @ . @ . @ . . @ . . . . . . . . @ . . . @ . . . . . . . . . . @
. . . @ . . @ . . @ . . . . . . . . . @ @ . . . . . @ . . . . @ . . . . . . . . . @ . . @ . . . @ @ @ . . . . . . . . . . . @ .
. . @ . . . . . @ . . . @ . . . . . . . . . . . . @ . . @ . . . . . . @ . . . . @ . . . . . . @ . . . . . @ @ @ . . . . @ . @ .
. . . . . . . . @ . . . . . @ @ . . . . @ . . . . . @ . @ . . . . . @ . . . . . @ . @ . . @ . . . . . . . . . . . . . . @ . . @
. . . @ . . . . . . . . . @ . . . . @ . . . . . . . . . . . . . . . . @ . @ @ . . . . . . . . . . . @ . . . . . @ . . . @ . . .
. . @ @ @ . . . . . . @ @ . . . . . . . . . . @ @ . . . . . . . . . . . . . . . . . @ . . @ . . . @ @ . . @ . . . . . . . @ . .
@ . . @ . . . . . . . . @ . . @ . @ . . . @ . . . . @ . . . . . . . @ . . . . . . @ . . . @ . . . . . . . . . . . @ . . . . . .
. . . . . @ . @ . . . . . . . @ . @ @ . . . . . . @ . . . . . @ . . @ . . . @ . . . . . . @ @ @ . . . . . @ . . . . . . . @ @ .
. . @ . . . . @ . . . @ @ . . . . @ . @ . @ @ @ . . . . . . . . @ . . . @ . @ . . . . . . . @ . . . . . @ . . . . . @ . . . . .
. . . . . . . . . . . . . @ @ . . . . . @ @ . . . @ . . . . . . . @ @ . @ . . . . @ . @ . . . . . @ @ . @ @ . . . . . . . @ . .
. . @ . . . . . . . @ . . . . . . @ . . . . . . @ . . . . . . . . . . . @ @ . @ . . . . . . . . . . @ . @ @ . . @ . . @ . . . .
. . . . . @ . . . . @ @ . . @ . . . @ . . . @ . . . . . . . @ . . . . . . @ . . . . @ . . @ @ . . . . @ . . . @ . @ . . . . . .
. . @ . . . . . . . @ . . @ . . . . . . . . . . . . . @ . @ @ . . . . . . @ @ . @ . . . @ . . . . . . . . . @ . . . . . . . . @
@ @ . . . @ . . . . @ . @ . . @ . @ . . @ . . @ . . @ . @ . @ . . . @ @ @ . . . @ . @ . . . . . . @ . . @ . . . . @ . . . . @ .
. . . . . @ . . . . . . . @ . . . . . . . @ . . @ . . . . . . . . . . @ . @ . . . . . . . . . . . . @ . . . . . @ . @ . @ . . .
. . . @ . . . . @ . . . . . . @ . @ . . . @ . . @ . . @ . . . . . @ @ @ . @ . . . @ . . . . . @ . . @ . @ . . @ . . . @ . . . .
. . . . . @ . . . . . @ . . . @ . . . @ @ . . . @ . . . @ @ . . @ . . . @ . . . . . . . . . . . . . . @ . . . @ . . . @ . @ @ .
. . . . . . . @ . . . . @ . @ @ . . . @ @ . @ @ . . . . . . . . @ . . . . @ . @ . @ @ @ . . @ . . . . @ . . . . @ @ . . @ . . .
. . . . . . @ . . . . @ @ . . . . @ @ . @ . @ . . . . @ . @ . @ . . @ . @ . . . . . . @ . . . @ . . . . . @ @ . . . . . . . . .
. . . . . . . @ @ . . @ . . . . . . . . . . . @ . . . . @ . . . . @ . . . . . . @ . . @ . . @ . . . @ . @ . @ @ @ . . . . . . @
. @ . . . . . . . . . @ . . . . @ . . @ . . . . . @ . . . . . . . . . @ @ . . @ . . . . . @ . . . . . @ . . . @ . . @ . . . . .
@ . @ . . . @ . . . @ . . . @ . . . @ @ @ @ @ . @ . . @ @ . . . . . . . . . . . . . . . . . . . . @ . @ @ . . . @ . . . . @ . .
@ . . . . . . . . . . @ @ . . . . . @ . . @ . . . @ . . . . . . . . @ @ . @ . @ . . . . @ . . @ @ . . . . . . . . . . . . . . .
@ . @ . . . . @ . . . . . . . . . . . . . . . @ . . . . . . . . @ . . @ . @ @ @ . . . . . . . . . . . . . @ @ . @ . . . . . . .
"""

tasks_data = """
50
7 27 0 24
14 47 47 49
63 5 61 59
3 53 29 23
28 2 22 56
42 49 60 2
21 8 34 5
26 13 20 54
47 33 34 10
22 50 61 33
23 32 40 61
1 49 12 41
55 45 48 15
43 6 44 30
41 1 21 57
57 5 47 1
16 61 2 11
20 50 45 56
9 54 47 56
12 31 62 22
40 51 6 53
40 40 42 33
52 36 26 34
42 48 28 26
55 32 23 46
5 41 18 24
8 62 18 63
56 49 4 59
39 30 52 3
5 0 50 28
29 27 59 9
29 50 18 28
21 63 20 12
16 29 30 40
36 18 60 21
50 18 19 40
12 36 8 23
45 36 43 39
35 52 53 54
49 28 55 5
24 40 25 9
15 25 62 62
17 52 4 34
32 43 14 50
63 9 36 57
8 46 44 0
31 43 55 14
59 1 12 11
15 1 60 8
56 0 0 63
"""

# 计算每个机器人的路径
grid = parse_map(map_data)
tasks = parse_tasks(tasks_data)

paths = []
total_time = 0
total_steps = 0  # 用于计算复杂度
branching_factor = 5  # 在每个节点处，最大可扩展5个方向（上、下、左、右、等待）

for i, task in enumerate(tasks):
    start, goal = task
    other_paths = [p[0] for p in paths]  # 其他机器人路径
    path, time = time_a_star_search(grid, start, goal, other_paths)
    paths.append((path, time))
    total_time = max(total_time, time)
    total_steps += len(path)

# 计算算法复杂度
grid_size = grid.shape[0] * grid.shape[1]
estimated_complexity = branching_factor ** (total_steps / len(tasks))

# 绘图
fig, ax = plt.subplots(figsize=(8, 8))
colors = plt.cm.get_cmap('tab20', len(tasks)).colors  # 使用更多颜色
step_map = np.full((grid.shape[0], grid.shape[1]), -1)
color_map_idx = np.full((grid.shape[0], grid.shape[1]), -1)

# 绘制障碍物
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        if grid[i, j] == '@':
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color='gray'))

# 绘制路径
for idx, (path, _) in enumerate(paths):
    color = colors[idx % len(colors)]
    if path:
        # 起点
        start_x, start_y = path[0]
        ax.add_patch(Circle((start_y + 0.5, start_x + 0.5), 0.4, edgecolor='black', facecolor=color, lw=2))

        # 终点
        end_x, end_y = path[-1]
        ax.add_patch(
            RegularPolygon((end_y + 0.5, end_x + 0.5), numVertices=6, radius=0.4, edgecolor='black', facecolor=color,
                           lw=2))

        # 路径及时间标记
        for t, (x, y) in enumerate(path):
            if step_map[x, y] < t + 1:
                step_map[x, y] = t + 1
                color_map_idx[x, y] = idx
                ax.add_patch(plt.Rectangle((y, x), 1, 1, color=color, alpha=0.5))

# 在方块上标记时间
for i in range(step_map.shape[0]):
    for j in range(step_map.shape[1]):
        if step_map[i, j] > 0:
            ax.text(j + 0.5, i + 0.5, str(step_map[i, j]-1), ha='center', va='center', color='black')

ax.set_xticks(np.arange(0, grid.shape[1] + 1, 1))
ax.set_yticks(np.arange(0, grid.shape[0] + 1, 1))
ax.grid(which='both', color='black', linestyle='-', linewidth=1)
plt.xlim(0, grid.shape[1])
plt.ylim(0, grid.shape[0])
plt.gca().invert_yaxis()  # 上下翻转y轴以匹配要求
plt.show()

# 输出结果
print(f"总运输时间: {total_time}")
print(f"估计的算法复杂度: O({estimated_complexity:.2e})")
for i, (path, time) in enumerate(paths, 1):
    print(f"机器人{i}: 路径 {path}, 时间消耗 {time}步")

if os.path.exists("results_第一问 64.xlsx"):
    # 清空文件内容
    with open("results_第一问 64.xlsx", 'w') as file:
        pass  # 使用 pass 清空文件内容

data = {
    "机器人编号": [],
    "位置列表": [],
    "时间开销": []
}

for i, (path, time) in enumerate(paths, 1):
    data["机器人编号"].append(i)
    data["位置列表"].append(path)
    data["时间开销"].append(time)

df = pd.DataFrame(data)

# 导出到Excel文件
df.to_excel('results_第一问 64.xlsx', index=False)
