import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
import scipy.integrate as integrate

eps = 1e-8
epsangle = 0.00001 * np.pi
maxangle = 32 * np.pi
cnt = 233
length0 = 2.86
length1 = 1.65
a = 0.55 / (2 * np.pi)

def r0(angle):
    return a * angle
def r1():
    return a
def s(x):
    return np.sqrt(r0(x) ** 2 + r1() ** 2)
def get_s(pre, now):
    ss = integrate.quad(s, pre, now)
    return ss[0]
def get_angle(t):
    l = 0.0
    r = 32.0 * np.pi
    while r - l > eps:
        mid = (l + r) / 2.0
        if get_s(mid, maxangle) <= t:r = mid
        else:l = mid
    return l
def get_x(angle):
    ans = r0(angle) * np.cos(angle)
    return ans
def get_y(angle):
    ans = r0(angle) * np.sin(angle)
    return ans
def get_xy(angle):
    return [get_x(angle), get_y(angle)]
def get_dis(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
def get_next_angle(x, len):
    def check(pre, now, len):
        pre_x = get_x(pre)
        pre_y = get_y(pre)
        now_x = get_x(now)
        now_y = get_y(now)
        return get_dis(pre_x, pre_y, now_x, now_y) <= len
    l = x
    r = x + np.pi / 2
    while r - l > eps:
        mid = (l + r) / 2.0
        if check(x, mid, len):l = mid
        else:r = mid
    return l
def get_next_v(pre, now, prev, len):
    pre2 = pre - epsangle
    now2 = get_next_angle(pre2, len)
    s_pre = get_s(pre2, pre)
    s_now = get_s(now2, now)
    return prev * s_now / s_pre

def visualization(pos, v):
    if len(pos)!= 0:
        print(len(pos))
        print(pos)
        x_coords = [p[0] for p in pos]
        y_coords = [p[1] for p in pos]

        plt.figure(figsize=(8, 8))
        plt.plot(x_coords, y_coords, marker='o')
        plt.title('Position Visualization')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.axis('equal')
        plt.show()

    if len(v)!= 0:
        print(len(v))
        print(v)
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(v)), v, marker='o', linestyle='-', color='b', label='Velocity (v)')
        plt.xlabel('Index')
        plt.ylabel('Velocity')
        plt.title('Velocity Change in Iteration 299')
        plt.legend()
        plt.grid(True)
        plt.show()
def solve(t, t1, step = 1, flag = 0):
    pos = []
    pos2 = []
    v = []
    angle = get_angle(t)
    prev = 1.0
    pos.append(get_x(angle))
    pos.append(get_y(angle))
    pos2.append(get_xy(angle))
    if flag: v.append(1.0)
    for i in range(cnt):
        angle2 = 0
        v2 = 0
        if i == 0:
            angle2 = get_next_angle(angle, length0)
            # if(angle2 > maxangle):break
            if flag:
                v2 = get_next_v(angle, angle2, prev, length0)
                v.append(v2)
        else:
            angle2 = get_next_angle(angle, length1)
            # if(angle2 > maxangle):break
            if flag:
                v2 = get_next_v(angle, angle2, prev, length1)
                v.append(v2)
        pos.append(get_x(angle2))
        pos.append(get_y(angle2))
        pos2.append(get_xy(angle2))
        angle = angle2
        prev = v2
    # visualization(pos2, v)
    return pos2, v
def GetCross(x1, y1, x2, y2, x, y):
    return (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)
def is_collision(x1, y1, x2, y2, x3, y3, x4, y4, x, y):
    return (GetCross(x1, y1, x2, y2, x, y) * GetCross(x3, y3, x4, y4, x, y) >= 0 and
            GetCross(x2, y2, x3, y3, x, y) * GetCross(x4, y4, x1, y1, x, y) >= 0)
def sort_counterclockwise(x1, y1, x2, y2, x3, y3, x4, y4):
    points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    center_x = (x1 + x2 + x3 + x4) / 4.0
    center_y = (y1 + y2 + y3 + y4) / 4.0
    def angle_from_center(point):
        x, y = point
        return math.atan2(y - center_y, x - center_x)
    points_sorted = sorted(points, key=angle_from_center)
    return points_sorted[0][0], points_sorted[0][1], points_sorted[1][0], points_sorted[1][1], \
        points_sorted[2][0], points_sorted[2][1], points_sorted[3][0], points_sorted[3][1]
def is_rect_collision(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8):
    x1, y1, x2, y2, x3, y3, x4, y4 = sort_counterclockwise(x1, y1, x2, y2, x3, y3, x4, y4)
    x5, y5, x6, y6, x7, y7, x8, y8 = sort_counterclockwise(x5, y5, x6, y6, x7, y7, x8, y8)
    return (is_collision(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5) or
                is_collision(x1, y1, x2, y2, x3, y3, x4, y4, x6, y6) or
                is_collision(x1, y1, x2, y2, x3, y3, x4, y4, x7, y7) or
                is_collision(x1, y1, x2, y2, x3, y3, x4, y4, x8, y8) or
                is_collision(x5, y5, x6, y6, x7, y7, x8, y8, x1, y1) or
                is_collision(x5, y5, x6, y6, x7, y7, x8, y8, x2, y2) or
                is_collision(x5, y5, x6, y6, x7, y7, x8, y8, x3, y3) or
                is_collision(x5, y5, x6, y6, x7, y7, x8, y8, x4, y4))
def get_xxyy(pos, i):
    x1 = pos[i][0]
    y1 = pos[i][1]
    x2 = pos[i + 1][0]
    y2 = pos[i + 1][1]
    if x1 > x2: x1, y1, x2, y2 = x2, y2, x1, y1
    return x1, y1, x2, y2
def get_rectangle(xx1, yy1, xx2, yy2, len):
    lenx = 0.15 * (yy2 - yy1) / len
    leny = 0.15 * (xx2 - xx1) / len
    xx3 = xx1 + (xx2 - xx1) * (len + 0.275) / len
    yy3 = yy1 + (yy2 - yy1) * (len + 0.275) / len
    x1 = xx3 + lenx
    x2 = xx3 - lenx
    y1 = yy3 - leny
    y2 = yy3 + leny
    xx4 = xx2 + (xx1 - xx2) * (len + 0.275) / len
    yy4 = yy2 + (yy1 - yy2) * (len + 0.275) / len
    x3 = xx4 - lenx
    x4 = xx4 + lenx
    y3 = yy4 + leny
    y4 = yy4 - leny
    return x1, y1, x2, y2, x3, y3, x4, y4
def plot_rectangles(rects):
    fig, ax = plt.subplots()
    for rect in rects:
        x1, y1, x2, y2, x3, y3, x4, y4 = rect
        polygon = Polygon([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], closed=True, edgecolor='r', fill=False)
        ax.add_patch(polygon)

    ax.set_aspect('equal')
    plt.xlim([-10, 10])  # Adjust limits based on expected coordinates
    plt.ylim([-10, 10])
    plt.grid(False)
    plt.title('Rectangles Visualization')
    plt.show()
def is_t_collision(t):
    pos, v = solve(t, t + 1)
    xx1, yy1, xx2, yy2 = get_xxyy(pos, 0)
    x1, y1, x2, y2, x3, y3, x4, y4 = get_rectangle(xx1, yy1, xx2, yy2, length0)
    xx11, yy11, xx21, yy21 = get_xxyy(pos, 1)
    x11, y11, x21, y21, x31, y31, x41, y41 = get_rectangle(xx11, yy11, xx21, yy21, length1)

    for i in range(2, 60):
        xx3, yy3, xx4, yy4 = get_xxyy(pos, i)
        x5, y5, x6, y6, x7, y7, x8, y8 = get_rectangle(xx3, yy3, xx4, yy4, length1)
        if i == 2:
            if (is_rect_collision(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8)):
                return True
        else:
            if (is_rect_collision(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8) or
                    is_rect_collision(x11, y11, x21, y21, x31, y31, x41, y41, x5, y5, x6, y6, x7, y7, x8, y8)):
                return True

def get_term_t(flag = 0):
    def get_term_t1(start, end, step):
        for t in np.arange(start, end, step):
            if is_t_collision(t): return t
    t = get_term_t1(0, 500,  1.0)
    step = 1.0
    for i in range(9):
        t = get_term_t1(t - step, t + step, step / 10)
        step = step / 10

    rects = []
    pos, v = solve(t, t + 1, 1, 1)
    xx1, yy1, xx2, yy2 = get_xxyy(pos, 0)
    rects.append(get_rectangle(xx1, yy1, xx2, yy2, length0))
    for i in range(1, cnt):
        xx3, yy3, xx4, yy4 = get_xxyy(pos, i)
        rects.append(get_rectangle(xx3, yy3, xx4, yy4, length1))
    if flag:
        plot_rectangles(rects)
    return t

def geta(x):
    global a
    a = x / (2 * np.pi)
def get_pitch():
    def check(x, flag = 0):
        geta(x)
        t = get_term_t(flag)
        return r0(get_angle(t)) <= 4.5
    l = 0.3
    r = 0.55
    while r - l > eps:
        mid = (l + r) / 2.0
        print(mid)
        if check(mid):r = mid
        else:l = mid
    print(l)
    check(l, 1)

#get_pitch()

def get2():
    geta(0.43)
    tt = []
    for t in np.arange(100, 300, 0.1):
        if is_t_collision(t):
            tt.append(t)
            print(len(tt))
    return tt

tt = get2()

# Prepare visualization.
t_values = np.arange(100, 300, 0.1)
y_values = np.zeros_like(t_values)

# Set y = 1 for values in tt.
for t in tt:
    index = np.where(np.isclose(t_values, t, atol=1e-3))[0]
    if index.size > 0:
        y_values[index[0]] = 1

# Plot the results.
plt.figure(figsize=(10, 4))
plt.plot(t_values, y_values, 'bo', label="Collision Points",markersize=1)
plt.title("Collision Points Visualization")
plt.xlabel("t (time)")
plt.ylabel("y")
plt.ylim(-0.1, 1.1)  # Set y limits for clarity
plt.grid(True)
plt.legend()
plt.show()