import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

path = kagglehub.dataset_download("mnassrib/telecom-churn-datasets")
path = f"{path}/churn-bigml-80.csv"
df = pd.read_csv(path)

# 1. Data Overview and Preprocessing
print(df.head())
print(df.info())
print(df.describe())

# Convert to numerical
df['International plan'] = df['International plan'].map({'No': 0, 'Yes': 1})
df['Voice mail plan'] = df['Voice mail plan'].map({'No': 0, 'Yes': 1})
df["Churn"] = df["Churn"].map({False: 0, True: 1})

# 2. Exploratory Data Analysis (EDA)
df = df.drop(['State'], axis=1)

# 3. Feature Engineering
df['Total charges'] = df['Total day charge'] + df['Total eve charge'] + df['Total night charge'] + df['Total intl charge']
df = df.drop(['Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge'], axis=1)

# One-hot encoding for categorical features
df = pd.get_dummies(df, columns=['Area code'], drop_first=True)

X = df.drop('Churn', axis=1)
y = df['Churn']

# Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Optimized Random Forest Model
model = RandomForestClassifier(random_state=42,
                               max_depth=5,
                               min_samples_leaf=10,
                               n_estimators=100,
                               class_weight='balanced')
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluation function
def evaluate_metrics(y_true, y_pred, dataset="Test"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"{dataset} Set Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")

evaluate_metrics(y_train, y_pred_train, "Train")
evaluate_metrics(y_test, y_pred_test, "Test")

# ---------------------------
# График метрик по порогам
# ---------------------------
y_scores = model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.0, 1.01, 0.01)
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

for t in thresholds:
    y_pred_threshold = (y_scores >= t).astype(int)
    accuracy_list.append(accuracy_score(y_test, y_pred_threshold))
    precision_list.append(precision_score(y_test, y_pred_threshold, zero_division=0))
    recall_list.append(recall_score(y_test, y_pred_threshold))
    f1_list.append(f1_score(y_test, y_pred_threshold))

plt.figure(figsize=(10, 6))
plt.plot(thresholds, accuracy_list, label='Accuracy')
plt.plot(thresholds, precision_list, label='Precision')
plt.plot(thresholds, recall_list, label='Recall')
plt.plot(thresholds, f1_list, label='F1-score')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Метрики в зависимости от порога классификации')
plt.legend()
plt.grid()
plt.show()

# ---------------------------
# Визуализации
# ---------------------------
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

sns.countplot(x='International plan', hue='Churn', data=df)
plt.title("Churn vs. International Plan")
plt.show()

sns.countplot(x='Customer service calls', hue='Churn', data=df)
plt.title("Churn vs. Customer Service Calls")
plt.show()

# Correlation Heatmap
corr_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(10, 8))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

save_dir = os.path.dirname(os.path.abspath(__file__)) 

df.to_csv(os.path.join(save_dir, "processed_churn_data.csv"), index=False)

metrics_df = pd.DataFrame({
    'Threshold': thresholds,
    'Accuracy': accuracy_list,
    'Precision': precision_list,
    'Recall': recall_list,
    'F1-score': f1_list
})

metrics_df.to_csv(os.path.join(save_dir, "threshold_metrics.csv"), index=False)

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

feature_importance_df.to_csv(os.path.join(save_dir, "feature_importances.csv"), index=False)

predictions_df = pd.DataFrame({
    'True Label': y_test.values,
    'Predicted Label': y_pred_test,
    'Predicted Probability': y_scores
})

predictions_df.to_csv(os.path.join(save_dir, "model_predictions.csv"), index=False)