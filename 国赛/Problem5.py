import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
import scipy.integrate as integrate

eps = 1e-12
eps2 = 1e-9
epsangle = 0.00001 * np.pi
maxangle = 32 * np.pi
cnt = 25
length0 = 2.86
length1 = 1.65
a = 0.55 / (2 * np.pi)
vv = 1
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
def get_dis(x1, y1, x2 = 0, y2 = 0):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
def get_s2(x1, y1, x2, y2, xo, yo, r):
    return r * math.acos(((x1 - xo) * (x2 - xo) + (y1 - yo) * (y2 - yo)) / (r * r))
def rotate(x1, y1, x2, y2, angle):
    x = (x1 - x2) * np.cos(angle) - (y1 - y2) * np.sin(angle) + x2
    y = (y1 - y2) * np.cos(angle) + (x1 - x2) * np.sin(angle) + y2
    return x, y
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
    prev = vv
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

geta(1.7)
angle = 4.5 / a
t0 = get_s(angle, maxangle)
def solve4(t, t_0):#t秒开始圆周运动，求t_0秒的位置
    angle2 = get_angle(t)
    x1 = get_x(angle2)
    y1 = get_y(angle2)
    x2 = -x1
    y2 = -y1
    x3 = (x1 + 2 * x2) / 3
    y3 = (y1 + 2 * y2) / 3
    d = 2 * r0(angle2)

    def geto():
        dy = np.sin(angle2) + angle2 * np.cos(angle2)
        dx = np.cos(angle2) - angle2 * np.sin(angle2)
        def f(x):
            return -dx / dy * (x - x1) + y1
        l = 0
        r = d
        while r - l > eps:
            mid = (l + r) / 2
            if abs(dy) <= eps:
                if get_dis(x1, y1 + mid, x1, y1) <= get_dis(x1, y1 + mid, x3, y3):l = mid
                else:r = mid
            else:
                if get_dis(x1 + mid, f(x1 + mid), x1, y1) <= get_dis(x1 + mid, f(x1 + mid), x3, y3) :l = mid
                else:r = mid
        if abs(dy) <= eps:
            if abs(get_dis(x1, y1 + l, x1, y1) - get_dis(x1, y1 + l, x3, y3)) <= eps2:
                return x1, y1 + l, -x1, -(y1 + l + y1) / 2
        else:
            if abs(get_dis(x1 + l, f(x1 + l), x1, y1) - get_dis(x1 + l, f(x1 + l), x3, y3)) <= eps2:
                return x1 + l, f(x1 + l), -(x1 + l + x1) / 2, -(f(x1 + l) + y1) / 2
        l = 0
        r = -d
        while r - l > eps:
            mid = (l + r) / 2
            if abs(dy) <= eps:
                if get_dis(x1, y1 + mid, x1, y1) <= get_dis(x1, y1 + mid, x3, y3):l = mid
                else:r = mid
            else:
                if get_dis(x1 + mid, f(x1 + mid), x1, y1) <= get_dis(x1 + mid, f(x1 + mid), x3, y3):l = mid
                else:r = mid
        if abs(dy) <= eps:
            if abs(get_dis(x1, y1 + l, x1, y1) - get_dis(x1, y1 + l, x3, y3)) <= eps2:
                return x1, y1 + l, -x1, -(y1 + l + y1) / 2
        else:
            if abs(get_dis(x1 + l, f(x1 + l), x1, y1) - get_dis(x1 + l, f(x1 + l), x3, y3)) <= eps2:
                return x1 + l, f(x1 + l), -(x1 + l + x1) / 2, -(f(x1 + l) + y1) / 2
        return 10000, 10000, 10000, 10000

    xo1, yo1, xo2, yo2 = geto()
    if(xo1 == yo1 == xo2 == yo2 == 10000):return [], [], 1

    ro1 = get_dis(xo1, yo1, x1, y1)
    ro2 = get_dis(xo2, yo2, x2, y2)
    so1 = get_s2(x1, y1, x3, y3, xo1, yo1, ro1)
    so2 = get_s2(x2, y2, x3, y3, xo2, yo2, ro2)
    def get_pos(tt, v):
        if tt <= t:
            return get_x(get_angle(tt)), get_y(get_angle(tt))
        tt -= t
        s = tt * v
        if s <= so1:
            return rotate(x1, y1, xo1, yo1, -s / ro1)
        elif s <= so1 + so2:
            s -= so1
            return rotate(x3, y3, xo2, yo2, s / ro2)
        else:
            s -= so1 + so2
            ttt = s / v
            angle3 = get_angle(t - ttt)
            return -get_x(angle3), -get_y(angle3)

    x_pre, y_pre = get_pos(t_0, 1.0)
    t_pre = t_0
    v_pre = vv
    pos = []
    v = []
    pos.append([x_pre, y_pre])
    v.append(v_pre)
    def get_nowpos(xx, yy, tt, len):
        def check(xx, yy, tt, len):
            xx2, yy2 = get_pos(tt, 1.0)
            return (get_dis(xx, yy, xx2, yy2) <= len)
        l = 0
        r = 3
        while r - l > eps:
            mid = (l + r) / 2.0
            if check(xx, yy, tt - mid, len):l = mid
            else:r = mid
        xx2, yy2 = get_pos(tt - l, 1.0)
        return xx2, yy2, tt - l

    def get_nowv(x_pre, y_pre, t_pre, v_pre, x_now, y_now, t_now, len):
        epslen = 0.00001
        x2_pre, y2_pre, t2_pre = get_nowpos(x_pre, y_pre, t_pre, epslen)
        x2_now, y2_now, t2_now = get_nowpos(x2_pre, y2_pre, t2_pre, len)
        if t2_now > t_now: return -1
        return get_dis(x2_now, y2_now, x_now, y_now) / epslen * v_pre

    for i in range(cnt):
        if i == 0:
            x_now, y_now, t_now = get_nowpos(x_pre, y_pre, t_pre, length0)
            v_now = get_nowv(x_pre, y_pre, t_pre, v_pre, x_now, y_now, t_now, length0)
            if v_now < -eps:return pos, v, 1
            v.append(v_now)
            v_pre = v_now
        else:
            x_now, y_now, t_now = get_nowpos(x_pre, y_pre, t_pre, length1)
            v_now = get_nowv(x_pre, y_pre, t_pre, v_pre, x_now, y_now, t_now, length1)
            if v_now < -eps:return pos, v, 1
            v.append(v_now)
            v_pre = v_now
        x_pre, y_pre, t_pre = x_now, y_now, t_now
        pos.append([x_pre, y_pre])

    #visualization(pos, [])
    return pos, v, 0

def collision(pos):
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

def check(x):
    angle2 = get_angle(x)
    d = 2 * r0(angle2)
    for t in np.arange(x, x + d, 0.1):
        pos, v, flag = solve4(x, t)
        if flag == 1 or collision(pos):
            return False
    return True

def getx():
    l = 9 + t0
    r = 11 + t0
    while r - l > eps2:
        mid = (l + r) / 2
        if check(mid):l = mid
        else:r = mid
    return l

x = 1340.177307524383
pos, v, flag = solve4(x, x + 50)
visualization(pos, v)
def check(v0):
    global vv
    vv = v0
    for t in np.arange(x, x + 40, 0.1):
        pos, v, flag = solve4(x, t)
        print(v)
        if max(v) > 2:return False
    return True

l = 1
r = 2
while r - l > eps2:
    mid = (l + r) / 2
    print(mid)
    if check(mid):l = mid
    else:r = mid

print(l)

vv = 1.631786035373807
pos, v, flag = solve4(x, x + 50)

