import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.interpolate import interp1d

data = pd.read_csv('历史用电数据_单柜.csv')

data.head()

cn_data = data[data['system_type'] == 'CN']['data'].values[0].split(',')
sd_data = data[data['system_type'] == 'SD']['data'].values[0].split(',')

cn_values = list(map(float, cn_data))
sd_values = list(map(float, sd_data))

# 计算负荷
load = [sd - cn for sd, cn in zip(sd_values, cn_values)]

# 显示前几条计算结果
print(load[:100])

# 可视化负荷数据
plt.figure(figsize=(10, 6))
plt.plot(load, label='Load')
plt.title('Load Over Time')
plt.xlabel('Time Intervals (15 seconds)')
plt.ylabel('Load (SD - CN)')
plt.legend()
plt.grid(True)
plt.show()

# 将负荷数据每60个数值进行均值处理
window_size = 60
processed_load = [np.mean(load[i:i + window_size]) for i in range(0, len(load), window_size)]


# 可视化处理后的负荷数据
plt.figure(figsize=(10, 6))
plt.plot(processed_load, label='Processed Load (Averaged every 60 intervals)')
plt.title('Processed Load Over Time')
plt.xlabel('Time Intervals (60 intervals)')
plt.ylabel('Load (Averaged SD - CN)')
plt.legend()
plt.grid(True)
plt.show()

# 计算平滑后的数据的一次差分
processed_load_diff = np.diff(processed_load)

# 根据数据大小调整滞后数量
lags = min(20, len(processed_load_diff) // 2 - 1)

# 绘制ACF和PACF图，帮助选择p, d, q参数
plt.figure(figsize=(14, 6))

# ACF 图
plt.subplot(1, 2, 1)
plot_acf(processed_load_diff, lags=lags, ax=plt.gca())
plt.title('ACF of Differenced Processed Load')

# PACF 图
plt.subplot(1, 2, 2)
plot_pacf(processed_load_diff, lags=lags, ax=plt.gca())
plt.title('PACF of Differenced Processed Load')

plt.show()


# 使用SARIMA模型进行预测，设定季节性周期为60个步长（15分钟，每步15秒）
seasonal_period = 80  # 15分钟的季节性周期对应60个步长

# 构建SARIMA模型，使用p=2, d=1, q=2 和季节性参数 P=1, D=1, Q=1, S=60
model = SARIMAX(processed_load,
                order=(2, 1, 2),
                seasonal_order=(1, 1, 1, seasonal_period))

# 拟合模型
model_fit = model.fit()

# 预测未来20个时间点
forecast = model_fit.forecast(steps=96)

# 显示预测结果
forecast

forecast = model_fit.forecast(steps=96)

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(processed_load)), processed_load, label='Observed Load')
plt.plot(np.arange(len(processed_load), len(processed_load) + 96), forecast, label='Forecasted Load', color='red')
plt.title('Observed and Forecasted Load')
plt.xlabel('Time Intervals')
plt.ylabel('Load (Averaged SD - CN)')
plt.legend()
plt.grid(True)
plt.show()


window_size = 60
restored_forecast = np.repeat(forecast, window_size)


original_indices = np.arange(0, len(processed_load) * window_size, window_size)
forecast_indices = np.arange(len(processed_load) * window_size, len(processed_load) * window_size + len(restored_forecast))

interp_function = interp1d(forecast_indices[::window_size], forecast, kind='linear', fill_value="extrapolate")

interpolated_forecast = interp_function(forecast_indices)

plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(load)), load, label='Original Load')
plt.plot(forecast_indices, interpolated_forecast, label='Interpolated Forecasted Load', color='red')
plt.title('Observed and Interpolated Forecasted Load')
plt.xlabel('Time Intervals')
plt.ylabel('Load (SD - CN)')
plt.legend()
plt.grid(True)
plt.show()

processed_data = pd.DataFrame({
    'Original Load': load,
    'Interpolated Forecasted Load': np.concatenate([np.full(len(load) - len(interpolated_forecast), np.nan), interpolated_forecast])
})

processed_data.to_excel('processed_load.xlsx', index=False)
