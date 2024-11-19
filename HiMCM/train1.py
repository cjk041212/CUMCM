import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
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
result_df['Sheet3'] = -4 * (result_df['Sheet3'] - 0.5) ** 2 + 1
if 'Sheet4' in result_df.columns:
    result_df['Sheet4'] = result_df['Sheet4'].apply(lambda x: 100 if x > 75 else x)
if 'Sheet6' in result_df.columns:
    result_df['Sheet6'] = result_df['Sheet6'].apply(lambda x: 1 if x > 1 else x)

scaler = StandardScaler()
for col in result_df.columns:
    result_df[col] = scaler.fit_transform(result_df[[col]])

x_final = result_df.values
y = np.array(y)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_final)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(x_pca, y, test_size=0.3, random_state=42)
svm_pca_model = SVC(kernel='rbf', probability=True)
svm_pca_model.fit(X_train_pca, y_train_pca)

y_pred_pca = svm_pca_model.predict(X_test_pca)
print("\nClassification Report (PCA Data):")
print(classification_report(y_test_pca, y_pred_pca))
print("\nAccuracy Score (PCA Data):", accuracy_score(y_test_pca, y_pred_pca))

def plot_decision_boundary(X, y, model, title='Decision Boundary'):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

plot_decision_boundary(X_train_pca, y_train_pca, svm_pca_model, title='Decision Boundary')
