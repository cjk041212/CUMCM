import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel('xlsx2.xlsx')
non_maintenance_df = df.dropna(subset=['时间段'])
time_slots = [(int(time.split('-')[0].split(':')[0]), int(time.split('-')[1].split(':')[0]))
              for time in non_maintenance_df['时间段']]

df = pd.read_excel('xlsx3.xlsx')
df.set_index('Unnamed: 0', inplace=True)
df.index.name = 'Parameter'

time_points = {
    "上午 10:00": 10,
    "中午 12:00": 12,
    "下午 14:00": 14,
    "下午 16:00": 16,
    "晚上 20:00": 20
}
dates = ["2024 年 9 月 8 日", "2024 年 9 月 9 日", "2024 年 9 月 10 日"]
aa = []
bb = []
cc = []

def temperature_model(t, c, a, b):
    return a - b * np.cos(np.pi / 12 * t - c)

def f(t, T, C0):
    return C0 * np.exp(-(10 ** ((T - 25) / 5)) * t)

def fit(model, time, temperature, a, b):
    popt, _ = curve_fit(lambda t, c: model(t, c, a, b), time, temperature)
    return popt[0]

def predict_ab(a, b, current_time):
    years = np.arange(2014, 2014 + len(a))
    def plot_acf_graph(data, title):
        month_day = title.split('年')[1]
        lags = min(9, len(data) - 1)
        plt.rcParams["figure.figsize"] = (10, 6)
        plot_acf(data, lags=lags)
        plt.title(f'{month_day} 的自相关函数（ACF）图')
        plt.xticks(ticks=np.arange(len(data)), labels=years)
        plt.xlabel('年份')
        plt.show()

    plot_acf_graph(a, f'{current_time}的 a 值')
    plot_acf_graph(b, f'{current_time}的 b 值')

    def predict_arima(series, order):
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return forecast[0], model_fit

    a_predicted, model_a = predict_arima(a, order=(1, 1, 1))
    b_predicted, model_b = predict_arima(b, order=(1, 1, 1))

    print(f'预测的今年的{current_time} 的 a 值: {a_predicted:.2f}')
    print(f'预测的今年的{current_time} 的 b 值: {b_predicted:.2f}')

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(years, a, label='历史数据')
    plt.plot(np.append(years, 2014 + len(a)), np.append(a, a_predicted), 'r--', label='预测值')
    plt.xlabel('年份')
    plt.ylabel('a 值')
    plt.title(f'{current_time} 的 a 值的 ARIMA 预测')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(years, b, label='历史数据')
    plt.plot(np.append(years, 2014 + len(b)), np.append(b, b_predicted), 'r--', label='预测值')
    plt.xlabel('年份')
    plt.ylabel('b 值')
    plt.title(f'{current_time} 的 b 值的 ARIMA 预测')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return a_predicted,b_predicted

def solve(id,current_time):
    time = []
    temperatures = []
    a = []
    b = []
    c = []
    pa = []
    pb = []

    for i in range(0, len(df.columns), 3):
        cols_all = df.columns[i: i + 3]
        data_all = df[cols_all]
        max_temps_all = data_all.loc['最高气温'].values
        min_temps_all = data_all.loc['最低气温'].values
        mean_temps_all = (max_temps_all.sum() + min_temps_all.sum()) / 6

        cols = df.columns[i + id]
        data = df[cols]
        max_temps = data.loc['最高气温']
        min_temps = data.loc['最低气温']
        mean_temps = (max_temps + min_temps) / 2

        time.extend([12, 13, 14, 20, 21, 22,
                     0, 1, 2, 6, 7, 8])
        temperatures.extend([
            max_temps, max_temps, max_temps, mean_temps + 1, mean_temps, mean_temps - 1,
            min_temps, min_temps, min_temps, mean_temps - 1, mean_temps, mean_temps + 1
        ])
        a.extend([mean_temps_all] * 12)
        b.extend([((max_temps - min_temps) / 2)] * 12)
        pa.extend([mean_temps_all])
        pb.extend([((max_temps - min_temps) / 2)])

    time = np.array(time)
    temperatures = np.array(temperatures)
    a = np.array(a)
    b = np.array(b)
    pa = np.array(pa)
    pb = np.array(pb)

    predict_a, predict_b = predict_ab(pa, pb, current_time)
    aa.append(predict_a)
    bb.append(predict_b)

    for i in range(len(a)):
        c.append(fit(temperature_model, time, temperatures, a[i], b[i]))

    c_mean = np.mean(c)
    #print(f'拟合得到的统一c值: {c_mean:.2f}')
    cc.append(c_mean)

def predict_pool_temperatures():
    predictions = {}
    for i, date_str in enumerate(dates):
        day_prediction = {}
        for time_label, time in time_points.items():
            temperature = temperature_model(time, cc[i], aa[i], bb[i]) - 3
            if(temperature >= 35):temperature -= 3.5
            else:temperature -= 2
            day_prediction[time_label] = temperature
        predictions[date_str] = day_prediction
    return predictions

def simulate_chlorine_concentration(time_slots, id):
    C0 = 0.6
    chlorine_concentration = []
    chlorine_times = []
    current_time = time_slots[0][0]

    for i, (start, end) in enumerate(time_slots):
        if (i == 0 or i == 3 or i == 6 or i == 9):
            chlorine_times.append(start)
            C0 = 0.6
        tt = 0.001
        while current_time < end:
            Tem = temperature_model(current_time, cc[id], aa[id], bb[id]) - 3
            if Tem > 35:
                Tem -= 2
            else:
                Tem -= 3.5
            C = f(tt, Tem, C0)
            # print(current_time, Tem)
            chlorine_concentration.append((current_time, C))
            current_time += tt
            if i == 2 or i == 5 or i == 8:
                C0 = C
            else:
                if C <= 0.3:
                    chlorine_times.append(current_time)
                    C0 = 0.6
                else:
                    C0 = C

    times, concentrations = zip(*chlorine_concentration)
    plt.figure(figsize=(12, 6))
    plt.plot(times, concentrations, label='余氯浓度 (mg/L)')
    plt.axhline(y=0.3, color='r', linestyle='--', label='阈值 (0.3 mg/L)')
    plt.xlabel('时间 (小时)')
    plt.ylabel('余氯浓度 (mg/L)')
    plt.title(f'{dates[id]} 余氯浓度随时间变化')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f'{dates[id]} 加氯时间点: {chlorine_times}')

for i in range(0, 3):
    solve(i,dates[i])
    print(f"日期: {dates[i]}")
    print(f"predict_a = {aa[i]},predict_b = {bb[i]},predict_c = {cc[i]}")

predicted_temperatures = predict_pool_temperatures()

for date, temps in predicted_temperatures.items():
    print(f"日期: {date}")
    max_temp = max(temps.values())
    min_temp = min(temps.values())
    print(f"最高温度: {max_temp:.2f}°C")
    print(f"最低温度: {min_temp:.2f}°C")

    for time, temp in temps.items():
        print(f"{time}: {temp:.2f}°C")
    print()

for i in range(0, 3):
    simulate_chlorine_concentration(time_slots, i)
