import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel('xlsx2.xlsx')
non_maintenance_df = df.dropna(subset=['时间段'])
time_slots = [(int(time.split('-')[0].split(':')[0]), int(time.split('-')[1].split(':')[0]))
              for time in non_maintenance_df['时间段']]

A = 0.003648497349718996
B = 0.34939455820847437

def f(t, T, n, C0):
    return C0 * np.exp(-(A * n + B) * 10 ** ((T - 25) / 5) * t)
def temperature_model(t):
    predict_a = 26.118397177120908
    predict_b = 3.666764465966214
    predict_c = 0.4967543828609517
    return predict_a - predict_b * np.cos(np.pi / 12 * t - predict_c)

def simulate_chlorine_concentration(start, end, n, c, i):
    C0 = c
    chlorine_concentration = []
    chlorine_times = []
    current_time = start
    cnt = 0;
    tt = 0.001
    while current_time < end:
        T = temperature_model(current_time) - 3
        if T > 35:
            T -= 2
        else:
            T -= 3.5
        C0 = f(tt, T, n, C0)
        chlorine_concentration.append((current_time, C0))
        current_time += tt
        if i == 2 or i == 5 or i == 8:
            continue
        else:
            if C0 <= 0.3:
                chlorine_times.append(current_time)
                C0 = 0.6
                cnt = cnt + 1

    return cnt, chlorine_times, chlorine_concentration, C0;

c = 0.6
cc = 0
flag = 0
chlorine_times = []
chlorine_concentration = []

def display(chlorine_times, chlorine_concentration):
    times, concentrations = zip(*chlorine_concentration)
    plt.figure(figsize=(12, 6))
    plt.plot(times, concentrations, label='余氯浓度 (mg/L)')
    plt.axhline(y=0.3, color='r', linestyle='--', label='阈值 (0.3 mg/L)')
    plt.xlabel('时间 (小时)')
    plt.ylabel('余氯浓度 (mg/L)')
    plt.title(f'{start} 余氯浓度随时间变化')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f'{start} 加氯时间点: {chlorine_times}')

for i, (start, end) in enumerate(time_slots):
    if i == 2 or i == 5 or i == 8:
        flag, chlorine_times, chlorine_concentration, cc = simulate_chlorine_concentration(start, end, 0, c, i)
        print('最大人数 = 0')
        #display(chlorine_times, chlorine_concentration)
        continue;
    if (i == 0 or i == 3 or i == 6 or i == 9):
        c = 0.6
    l = 0
    r = 520
    while l < r:
        mid = int((l + r + 1) / 2)
        flag , chlorine_times, chlorine_concentration ,cc = simulate_chlorine_concentration(start, end, mid ,c ,i)
        if(flag <= 0):l = mid
        else: r = mid - 1
    c = cc
    #display(chlorine_times, chlorine_concentration)
    print(f'最大人数 = {l}')

