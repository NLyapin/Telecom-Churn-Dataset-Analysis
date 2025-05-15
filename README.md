Вот пример `README.md` для вашего проекта на основе анализа оттока клиентов с использованием моделей XGBoost и Random Forest:

---

# Customer Churn Prediction

Проект по прогнозированию оттока клиентов на основе данных телеком-компании с применением методов машинного обучения: **XGBoost** и **Random Forest**. Выполнена полная обработка данных, балансировка классов, визуализация, построение кривых метрик и сохранение результатов.

---

## Данные

Используется датасет с [Kaggle](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets): `churn-bigml-80.csv`.

Загрузка выполняется автоматически через `kagglehub`.

---

## Используемые технологии

* Python
* Pandas / NumPy
* Matplotlib / Seaborn
* Scikit-learn
* imbalanced-learn (SMOTE)
* XGBoost
* RandomForestClassifier

---

## Этапы обработки

1. **Загрузка данных**
2. **Предобработка**

   * Преобразование категориальных признаков (`map`, `get_dummies`)
   * Создание нового признака `Total charges`
   * Удаление нерелевантных признаков (`State`, `charges`)
3. **Нормализация** с помощью `StandardScaler`
4. **Разделение на train/test**
5. **Балансировка классов** через SMOTE (для XGBoost)
6. **Обучение моделей**

   * `XGBClassifier` с параметрами (`max_depth=3`, `n_estimators=50`, и т.д.)
   * `RandomForestClassifier` с параметрами (`max_depth=5`, `n_estimators=100`, и т.д.)
7. **Оценка метрик**:

   * Accuracy
   * Precision
   * Recall
   * F1-score
8. **Анализ метрик по порогам**
9. **Визуализация**:

   * Матрица корреляций
   * Важность признаков
   * Распределения классов
10. **Сохранение результатов** в `.csv`

---

## Результаты

* Метрики на трейне и тесте
* Визуализация важности признаков
* Кривые зависимости метрик от порогов
* Графики распределения Churn по различным признакам
* Сохранённые таблицы:

  * `processed_churn_data.csv`
  * `feature_importances.csv`
  * `threshold_metrics.csv`
  * `model_predictions.csv`

---

## 📦 Запуск

### Установка зависимостей

```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost kagglehub
```

### Запуск

```bash
python churn_analysis_xgboost.py
python churn_analysis_randomforest.py
```

*Файлы разделены по моделям. Каждый скрипт самодостаточен.*