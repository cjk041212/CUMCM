import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate

eps = 1e-12
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
def check3(pre, now, len):
    pre_x = get_x(pre)
    pre_y = get_y(pre)
    now_x = get_x(now)
    now_y = get_y(now)
    return get_dis(pre_x, pre_y, now_x, now_y) <= len
def get_next_angle(x, len):
    l = x
    r = x + np.pi / 2
    while r - l > eps:
        mid = (l + r) / 2.0
        if check3(x, mid, len):l = mid
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
def solve(t0, t1):
    for t in range(t0, t1):
        print(t)
        pos = []
        pos2 = []
        v = []
        angle = get_angle(t)
        prev = 1.0
        pos.append(get_x(angle))
        pos.append(get_y(angle))
        pos2.append(get_xy(angle))
        v.append(1.0)
        for i in range(cnt):
            angle2 = 0
            v2 = 0
            if i == 0:
                angle2 = get_next_angle(angle, length0)
                #if(angle2 > maxangle):break
                v2 = get_next_v(angle, angle2, prev, length0)
                v.append(v2)
            else:
                angle2 = get_next_angle(angle, length1)
                #if(angle2 > maxangle):break
                v2 = get_next_v(angle, angle2, prev, length1)
                v.append(v2)
            pos.append(get_x(angle2))
            pos.append(get_y(angle2))
            pos2.append(get_xy(angle2))
            angle = angle2
            prev = v2

        if(t == t1 - 1):
            visualization(pos2, v)

solve(0, 301)

