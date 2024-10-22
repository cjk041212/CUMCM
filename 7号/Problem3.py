import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

A = 0.003648497349718996
B = 0.34939455820847437
K = 0.46209812037329684


def f1(t, C0):
    return C0 * np.exp(-K * t)
def f2(t, n, C0):
    return C0 * np.exp(-0.5 * (A * n + B) * t)

df = pd.read_excel('xlsx2.xlsx')
#print(df)
non_maintenance_df = df.dropna(subset=['时间段'])
time_slots = [(int(time.split('-')[0].split(':')[0]), int(time.split('-')[1].split(':')[0]))
              for time in non_maintenance_df['时间段']]
n_values = non_maintenance_df['在池游泳人数平均统计'].tolist()

#print(time_slots)
#print(n_values)
#print(time_slots2)
#print(n_values2)

def simulate_chlorine_concentration(time_slots, n_values):
    C0 = 0.6
    chlorine_concentration = []
    chlorine_times = []
    current_time = time_slots[0][0]

    for i, (start, end) in enumerate(time_slots):
        n = n_values[i]
        if (i == 0 or i == 3 or i == 6 or i == 9):
            chlorine_times.append(start)
            C0 = 0.6
        tt=0.01
        while current_time < end:
            print(current_time)
            #if n == 0:
            #    C = f1(tt, C0)
            #else:
            C = f2(tt, n, C0)
            chlorine_concentration.append((current_time, C))
            current_time += tt
            if(i == 2 or i == 5 or i == 8):
                C0 = C
            else:
                if C <= 0.3:
                    chlorine_times.append(current_time)
                    C0 = 0.6
                else:
                    C0 = C

    return chlorine_concentration, chlorine_times


chlorine_concentration, chlorine_times = simulate_chlorine_concentration(time_slots, n_values)

times, concentrations = zip(*chlorine_concentration)
plt.figure(figsize=(12, 6))
plt.plot(times, concentrations, label='Chlorine concentration (mg/L)')
plt.axhline(y=0.3, color='r', linestyle='--', label='Threshold (0.3 mg/L)')
plt.xlabel('Time (hours)')
plt.ylabel('Chlorine concentration (mg/L)')
plt.title('Chlorine Concentration Over Time')
plt.legend()
plt.grid(True)
plt.show()

print(f'加氯时间点: {chlorine_times}')