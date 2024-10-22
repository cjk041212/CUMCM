import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found")

file_path = '1.xlsx'
data = pd.read_excel(file_path, usecols=["StockCode", "Quantity", "InvoiceDate"])
data['YearMonth'] = pd.to_datetime(data['InvoiceDate']).dt.to_period('M')
monthly_sales = data.groupby(['StockCode', 'YearMonth'])['Quantity'].sum().reset_index()
monthly_sales = monthly_sales.set_index(['StockCode', 'YearMonth']).unstack(fill_value=0).stack().reset_index()
monthly_sales = monthly_sales.sort_values(by=['StockCode', 'YearMonth'])
quantity_sequences = monthly_sales.groupby('StockCode')['Quantity'].apply(list).reset_index()

print(quantity_sequences)

def predi(quantity_sequence, x):
    best_stock_code = None
    best_forecast_stock_code = None
    max_forecast_value = -np.inf
    forecast_results = []

    for index, row in quantity_sequence.iterrows():
        stock_code = row['StockCode']
        quantity_series = pd.Series(row['Quantity'])

        if len(quantity_series) >= x:
            model = ARIMA(quantity_series[:x], order=(1, 1, 1))
            model_fit = model.fit()

            forecast = model_fit.forecast(steps=1)

            forecast_results.append({
                'StockCode': stock_code,
                'Forecast': int(max(forecast.tolist()[0], 0)),
            })

            if forecast.max() > max_forecast_value:
                max_forecast_value = forecast.max()
                best_stock_code = quantity_series[:x]
                best_forecast_stock_code = stock_code

    print(forecast_results)

    forecast_results_sorted = sorted(forecast_results, key=lambda x: x['Forecast'], reverse=True)
    top_10_forecast_results = forecast_results_sorted[:10]
    print("Top 10 forecasted values:")
    for result in top_10_forecast_results:
        print(f"StockCode: {result['StockCode']}, Forecast: {result['Forecast']}")

    if best_stock_code is not None:
        print(f"\nPlotting ACF for StockCode: {best_forecast_stock_code}")
        plt.figure(figsize=(10, 6))
        lags = min(20, len(best_stock_code) - 1)
        plot_acf(best_stock_code, lags=lags)
        plt.title(f"ACF for StockCode: {best_forecast_stock_code}")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(best_stock_code)), best_stock_code, marker='o', label='Actual')
        plt.scatter(len(best_stock_code), max_forecast_value, color='red', marker='x', s=100, label='Forecast')
        plt.title(f"Actual vs Forecasted Quantity for StockCode: {best_forecast_stock_code}")
        plt.xlabel('Time Period')
        plt.ylabel('Quantity')
        plt.legend()
        plt.grid(True)
        plt.show()

predi(quantity_sequences, 5)
predi(quantity_sequences, 13)
