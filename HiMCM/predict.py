import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from joblib import dump,load
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

file_path = 'aaa.xlsx'
file_path2 = 'bbb.xlsx'
sheet_names = pd.ExcelFile(file_path).sheet_names
sheet_data = []

for sheet_name in sheet_names:
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    sheet_data.append(data.iloc[:, 2:])
data = pd.read_excel(file_path2).iloc[:, 2:]

x = []
y = []
rows, cols = sheet_data[0].shape
for i in range(rows):
    for j in range(cols):
        vector = [sheet.iloc[i, j] for sheet in sheet_data]
        x.append(vector)
for i in range(rows):
    for j in range(cols):
        y_value = data.iloc[i, j]
        y.append(1 if y_value > 1 else y_value)


result_df = pd.DataFrame(x, columns=[f'Sheet{i + 1}' for i in range(len(sheet_data))])
result_df['ID'] = result_df.index + 1

result_df['Sheet3'] = -4 * (result_df['Sheet3'] - 0.5) ** 2 + 1
if 'Sheet4' in result_df.columns:
    result_df['Sheet4'] = result_df['Sheet4'].apply(lambda x: 100 if x > 75 else x)
if 'Sheet6' in result_df.columns:
    result_df['Sheet6'] = result_df['Sheet6'].apply(lambda x: 1 if x > 1 else x)

scaler = StandardScaler()
x_final = scaler.fit_transform(result_df.drop(columns=['ID']).values)
x_final_with_ids = np.hstack([result_df['ID'].values.reshape(-1, 1), x_final])

feature_weights = [30.0, 30.0, 16.02846270949682, 13.100611947884852, 1.0, 1.0]
x_weighted = x_final_with_ids[:, 1:] * feature_weights
X_train, X_test, y_train, y_test = train_test_split(x_weighted, y, test_size=0.3, random_state=42)
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, y_train)

dump(scaler, 'scaler.joblib')
dump(feature_weights, 'feature_weights.joblib')

def process_and_predict_svm_frozen(x_new, svm_model, scaler, feature_weights):
    for i in range(len(x_new)):
        x_new[i][2] = -4 * (x_new[i][2] - 0.5) ** 2 + 1
        x_new[i][3] = 100 if x_new[i][3] > 75 else x_new[i][3]
        x_new[i][5] = 1 if x_new[i][5] > 1 else x_new[i][5]
    x_new_scaled = scaler.transform(x_new)
    x_new_weighted = x_new_scaled * feature_weights
    predicted_prob = svm_model.predict_proba(x_new_weighted)[:, 1]
    predicted_labels = (predicted_prob > 0.5).astype(int)
    print("Predicted Probabilities:", predicted_prob)
    print("Predicted Labels:", predicted_labels)
    return predicted_prob, predicted_labels

svm_model = load('svm_model.joblib')
scaler = load('scaler.joblib')
feature_weights = load('feature_weights.joblib')

x_add = [[80, 4000, 0.5, 13, 0.046875, 0],
         [19, 11000, 0.5, 36, 0.037037037, 0],
         [14, 5000, 0, 9, 0.042735043, 0]]
#3v3,Karate,softball
x_exi = [[192, 5000, 0.5, 56, 0.295081967, 0],
         [380, 95000, 0.511781338, 190, 0.316831683, 1],
         [97, 5800, 0.504166667, 33, 0.330357143, 0]]
#Weightlifting,Athletics, Cycling Track

process_and_predict_svm_frozen(x_add, svm_model, scaler, feature_weights)
process_and_predict_svm_frozen(x_exi, svm_model, scaler, feature_weights)

data1 = [[43, 39, 38, 46, 48, 57, 50],
          [3500, 3500, 3500, 3500, 3500, 3500, 3500],
          [0.6, 0.58, 0.56, 0.55, 0.54, 0.54, 0.53],
          [13, 16, 20, 22, 22, 22, 24],
          [0.3, 0.34, 0.38, 0.39, 0.42, 0.45, 0.46]]#Croquet

data2 = [[103, 49, 46, 80, 45, 76, 153],
         [60000, 60000, 60000, 60000, 60000, 60000, 60000],
         [0.87, 0.85, 0.81, 0.76, 0.75, 0.72, 0.7],
         [8, 10, 11, 14, 15, 15, 18],
         [0.4, 0.43, 0.45, 0.47, 0.48, 0.52, 0.57]]#Polo

data3 =[[43, 40, 37, 54, 49, 64, 160],
        [4700, 4700, 4700, 4700, 4700, 4700, 4700],
        [0.484848485, 0.484848485, 0.484848485, 0.484848485, 0.484848485, 0.484848485, 0.484848485],
        [16, 16, 16, 16, 16, 16, 16],
        [0.393939394, 0.393939394, 0.393939394, 0.393939394, 0.393939394, 0.393939394, 0.393939394]]#breaking
data4 = [[50.3, 62.7, 105.4, 120.8, 175.2, 190.5, 230.7, 245.3, 290.1, 295.4, 298.7, 299.2, 299.8],
         [5000, 5000, 10000, 10000, 15000, 15000, 20000, 20000, 25000, 25000, 30000, 30000, 35000],
         [0.85, 0.84, 0.83, 0.82, 0.81, 0.80, 0.79, 0.78, 0.77, 0.76, 0.75, 0.74, 0.73],
         [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
         [0.70, 0.72, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84]]#electronic sports
data5 = [[17, 28, 27, 26, 39, 35, 32],
         [5000, 5000, 5000, 5000, 5000, 5000, 5000],
         [0.89, 0.87, 0.85, 0.84, 0.82, 0.82, 0.8],
         [8, 10, 12, 14, 16, 22, 26],
         [0.12, 0.14, 0.17, 0.173, 0.18, 0.193, 0.2]]#Basque Pelota
data6 = [[86, 51, 34, 69, 72, 73, 112],
         [25000, 25000, 25000, 25000, 25000, 25000, 25000],
         [0.95, 0.92, 0.86, 0.84, 0.82, 0.82, 0.8],
         [14, 16, 18, 22, 24, 25, 28],
         [0.3, 0.34, 0.365, 0.377, 0.381, 0.392, 0.4]]#Cricket

def predict2(data,step):
    def predict(data,step):
        model = ARIMA(data, order=(1, 1, 1))
        fitted_model = model.fit()
        forecast = fitted_model.get_forecast(steps=step)
        resid = fitted_model.resid
        max_lags = min(20, len(resid) - 1)  # 确保lags不超过数据长度
        plt.figure(figsize=(8, 5))
        plot_acf(resid, lags=max_lags, title=None)
        plt.show()
        return forecast.predicted_mean[-1]
    forecast_2032 = [predict(data[i],step) for i in range(5)]
    forecast_2032.append(0)
    print(forecast_2032)
    process_and_predict_svm_frozen([forecast_2032], svm_model, scaler, feature_weights)

#predict2(data1,2)
#predict2(data2,2)
#predict2(data3,2)
predict2(data4,8)
#predict2(data5,2)
#predict2(data6,2)

