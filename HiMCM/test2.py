import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from skopt import gp_minimize
from skopt.space import Real
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = 'aaa.xlsx'
file_path2 = 'bbb.xlsx'
sheet_names = pd.ExcelFile(file_path).sheet_names
sheet_data = []

# Read data from each sheet
for sheet_name in sheet_names:
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    sheet_data.append(data.iloc[:, 2:])
data = pd.read_excel(file_path2).iloc[:, 2:]

# Extract features (x) and target values (y)
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
# Create DataFrame with ID column
result_df = pd.DataFrame(x, columns=[f'Sheet{i + 1}' for i in range(len(sheet_data))])
result_df['ID'] = result_df.index + 1
print(result_df)

# Feature transformation
result_df['Sheet3'] = -4 * (result_df['Sheet3'] - 0.5) ** 2 + 1
if 'Sheet4' in result_df.columns:
    result_df['Sheet4'] = result_df['Sheet4'].apply(lambda x: 100 if x > 75 else x)
if 'Sheet6' in result_df.columns:
    result_df['Sheet6'] = result_df['Sheet6'].apply(lambda x: 1 if x > 1 else x)

# Standardize features
scaler = StandardScaler()
x_final = scaler.fit_transform(result_df.drop(columns=['ID']).values)
x_final_with_ids = np.hstack([result_df['ID'].values.reshape(-1, 1), x_final])


# Bayesian optimization function
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

    # Plot the loss function changes during Bayesian optimization
    plt.figure(figsize=(10, 6))
    plt.plot(result.func_vals, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Negative Accuracy (Loss)')
    plt.grid(True)
    plt.show()

    best_weights = result.x
    print(f"Best feature weights: {best_weights}")
    return np.array(best_weights)


# Perform Bayesian optimization to get the best feature weights
feature_weights = bayesian_optimization(x_final_with_ids, y)
x_weighted = x_final_with_ids[:, 1:] * feature_weights

# Cross-validation using StratifiedKFold
svm_model = SVC(kernel='rbf', probability=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cross_val_accuracy = []
fold = 1

# Perform cross-validation and plot confusion matrix for each fold
for train_index, test_index in skf.split(x_weighted, y):
    X_train, X_test = x_weighted[train_index], x_weighted[test_index]
    y_train, y_test = y[train_index], y[test_index]

    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    # Compute and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    # plt.title(f'Confusion Matrix for Fold {fold}')
    plt.show()

    accuracy = accuracy_score(y_test, y_pred)
    cross_val_accuracy.append(accuracy)
    fold += 1

# Output cross-validation results
print(f"Cross-validation accuracies: {cross_val_accuracy}")
print(f"Mean cross-validation accuracy: {np.mean(cross_val_accuracy):.4f}")
print(f"Standard deviation of cross-validation accuracy: {np.std(cross_val_accuracy):.4f}")

# Train final model and compute test set accuracy
X_train, X_test, y_train, y_test = train_test_split(x_weighted, y, test_size=0.3, random_state=42)
svm_model.fit(X_train, y_train)
y_test_pred = svm_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy Score: {test_accuracy:.4f}")


# 1. Plot feature importance
def plot_feature_importance(feature_weights, result_df):
    feature_names = result_df.columns[:-1]  # Exclude 'ID' column
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_weights)
    plt.xlabel('Feature Importance (Weight)')
    plt.ylabel('Feature')
    #plt.title('Feature Importance Based on Optimized Weights')
    plt.show()


plot_feature_importance(feature_weights, result_df)


# 2. Plot learning curve
def plot_learning_curve(model, X_train, y_train, title="Learning Curve"):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=StratifiedKFold(n_splits=5), n_jobs=-1
    )
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score', color='blue')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation Score', color='red')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Score')
    #plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


plot_learning_curve(svm_model, X_train, y_train)


# 3. Plot ROC curve
def plot_roc_curve(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]  # Get the probability for the positive class
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


plot_roc_curve(svm_model, X_test, y_test)


# 4. Plot feature distributions
def plot_feature_distributions(result_df):
    plt.figure(figsize=(15, 10))
    result_df.drop(columns=['ID']).hist(bins=30, figsize=(15, 10))
    #plt.suptitle('Feature Distributions')
    plt.show()


plot_feature_distributions(result_df)


# 5. Plot feature correlation heatmap
def plot_feature_correlation(result_df):
    correlation_matrix = result_df.drop(columns=['ID']).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    #plt.title('Feature Correlation Heatmap')
    plt.show()


plot_feature_correlation(result_df)


# 6. Plot prediction vs true values
def plot_prediction_vs_true(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='True Values', alpha=0.6)
    plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Values', alpha=0.6)
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    #plt.title('True Values vs Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_prediction_vs_true(y_test, y_test_pred)
