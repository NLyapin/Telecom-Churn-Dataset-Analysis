import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# качаем датасет
path = kagglehub.dataset_download("mnassrib/telecom-churn-datasets")
path = f"{path}/churn-bigml-80.csv"
df = pd.read_csv(path)

print(df.head())
print(df.info())
print(df.describe())

# категории в числа
df['International plan'] = df['International plan'].map({'No': 0, 'Yes': 1})
df['Voice mail plan'] = df['Voice mail plan'].map({'No': 0, 'Yes': 1})
df["Churn"] = df["Churn"].map({False: 0, True: 1})

# удаляем штат
df = df.drop(['State'], axis=1)

# создаём новый признак — сумма всех расходов
df['Total charges'] = df['Total day charge'] + df['Total eve charge'] + df['Total night charge'] + df['Total intl charge']
df = df.drop(['Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge'], axis=1)

# кодируем area code (категория)
df = pd.get_dummies(df, columns=['Area code'], drop_first=True)

# делим фичи и цель
X = df.drop('Churn', axis=1)
y = df['Churn']

# нормализуем признаки
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# делим на train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# обучаем RandomForest — настроенный под баланс классов
model = RandomForestClassifier(random_state=42,
                               max_depth=5,  # меньше глубина - меньше переобучения
                               min_samples_leaf=10,
                               n_estimators=100,
                               class_weight='balanced')  # балансим классы
model.fit(X_train, y_train)

# предсказываем
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# функция для оценки метрик
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

# метрики для трейна и теста
evaluate_metrics(y_train, y_pred_train, "Train")
evaluate_metrics(y_test, y_pred_test, "Test")

# визуализация
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

sns.countplot(x='International plan', hue='Churn', data=df)
plt.title("Churn vs. International Plan")
plt.show()

sns.countplot(x='Customer service calls', hue='Churn', data=df)
plt.title("Churn vs. Customer Service Calls")
plt.show()

corr_matrix = df.corr()
plt.figure(figsize=(13, 7))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

importances = model.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(10, 8))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()