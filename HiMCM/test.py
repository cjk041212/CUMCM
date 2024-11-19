import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from skopt import gp_minimize
from skopt.space import Real
import matplotlib.pyplot as plt

# 加载数据
file_path = 'aaa.xlsx'
file_path2 = 'bbb.xlsx'
sheet_names = pd.ExcelFile(file_path).sheet_names
sheet_data = []

# 读取各个sheet的数据
for sheet_name in sheet_names:
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    sheet_data.append(data.iloc[:, 2:])
data = pd.read_excel(file_path2).iloc[:, 2:]

# 提取特征（x）和目标值（y）
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

y = np.array(y)
# 构造DataFrame，添加ID列
result_df = pd.DataFrame(x, columns=[f'Sheet{i + 1}' for i in range(len(sheet_data))])
result_df['ID'] = result_df.index + 1
print(result_df)

# 特征转换
result_df['Sheet3'] = -4 * (result_df['Sheet3'] - 0.5) ** 2 + 1
if 'Sheet4' in result_df.columns:
    result_df['Sheet4'] = result_df['Sheet4'].apply(lambda x: 100 if x > 75 else x)
if 'Sheet6' in result_df.columns:
    result_df['Sheet6'] = result_df['Sheet6'].apply(lambda x: 1 if x > 1 else x)

# 标准化特征
scaler = StandardScaler()
x_final = scaler.fit_transform(result_df.drop(columns=['ID']).values)
x_final_with_ids = np.hstack([result_df['ID'].values.reshape(-1, 1), x_final])


# 贝叶斯优化函数
def optimize_feature_weights(weights, x_final, y, test_size=0.3, random_state=42):
    x_weighted = x_final[:, 1:] * weights
    X_train, X_test, y_train, y_test = train_test_split(x_weighted, y, test_size=test_size, random_state=random_state)
    svm_model = SVC(kernel='rbf')
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    return -accuracy_score(y_test, y_pred)


def bayesian_optimization(x_final, y, test_size=0.3, random_state=42):
    n_features = x_final.shape[1] - 1
    search_space = [Real(1, 30) for _ in range(n_features)]
    result = gp_minimize(lambda weights: optimize_feature_weights(weights, x_final, y, test_size, random_state),
                         dimensions=search_space,
                         n_calls=50,
                         random_state=random_state)
    plt.figure(figsize=(10, 6))
    plt.plot(result.func_vals, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Negative Accuracy (Loss)')
    plt.grid(True)
    plt.show()
    best_weights = result.x
    print(f"best feature_weights: {best_weights}")
    return np.array(best_weights)


# 贝叶斯优化获取最佳特征权重
feature_weights = bayesian_optimization(x_final_with_ids, y)
x_weighted = x_final_with_ids[:, 1:] * feature_weights

# 使用StratifiedKFold进行交叉验证
svm_model = SVC(kernel='rbf', probability=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cross_val_accuracy = []
fold = 1

# 循环进行交叉验证，并绘制每个折叠的混淆矩阵
for train_index, test_index in skf.split(x_weighted, y):
    X_train, X_test = x_weighted[train_index], x_weighted[test_index]
    y_train, y_test = y[train_index], y[test_index]

    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    # 计算并显示混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    #plt.title(f'Confusion Matrix for Fold {fold}')
    plt.show()

    accuracy = accuracy_score(y_test, y_pred)
    cross_val_accuracy.append(accuracy)
    fold += 1

# 输出交叉验证结果
print(f"Cross-validation accuracies: {cross_val_accuracy}")
print(f"Mean cross-validation accuracy: {np.mean(cross_val_accuracy):.4f}")
print(f"Standard deviation of cross-validation accuracy: {np.std(cross_val_accuracy):.4f}")

# 训练最终模型并计算测试集准确率
X_train, X_test, y_train, y_test = train_test_split(x_weighted, y, test_size=0.3, random_state=42)
svm_model.fit(X_train, y_train)
y_test_pred = svm_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy Score: {test_accuracy:.4f}")
