import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 1. Загрузка и подготовка данных
df = pd.read_excel('processed_ml_dataset_with_lmtd.xlsx', parse_dates=['datetime'])
df['hour_of_day'] = df['datetime'].dt.hour
df['time_since_start'] = (
    df['datetime'] - df.groupby('run')['datetime'].transform('min')
).dt.total_seconds() / 3600

# 2. Составление первоначального X и y
feature_cols = [
    'hour_of_day', 'time_since_start',
    'HE-1 Feed In', 'HE-1 Feed Out', 'HE-1 Prod In',
    'HE-1 Bend 1', 'HE-1 Bend 2', 'HE-1 Bend 3', 'HE-1 Bend 4',
    'HE-1 Shell 1', 'HE-1 Shell 2', 'HE-1 Shell 3', 'HE-1 Shell 4',
    'dTL_leg1', 'dTL_leg2', 'dTL_leg3', 'dTL_leg4', 'dTL_leg5',
    'feed_material', 'run'
]
df_model = df[feature_cols + ['target_prod_out']].dropna()
X = df_model[feature_cols]
y = df_model['target_prod_out']

# 3. Сплит для первой модели
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Тренируем начальную модель для оценки важности
cat_features = ['feed_material', 'run']
pool_train = Pool(X_train, y_train, cat_features=cat_features)
model_init = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric='RMSE',
    random_seed=42,
    verbose=False
)
model_init.fit(pool_train)

# 5. Извлекаем feature importances
importances = model_init.get_feature_importance(pool_train)
feat_names  = model_init.feature_names_
feat_imp_df = pd.DataFrame({
    'feature': feat_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# 6. Фильтрация признаков по порогу
threshold = 10.0
selected = feat_imp_df.loc[feat_imp_df['importance'] > threshold, 'feature'].tolist()

print("Выбранные признаки (importance >", threshold, "):")
print(selected)

# 7. Финальная модель на отобранных фичах
X_sel = X[selected]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_sel, y, test_size=0.2, random_state=42)
pool_train2 = Pool(X_train2, y_train2, cat_features=[f for f in ['feed_material','run'] if f in selected])
model_final = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric='RMSE',
    random_seed=42,
    verbose=False
)
model_final.fit(pool_train2)

# 8. Оценка финальной модели
y_pred = model_final.predict(X_test2)
mse  = mean_squared_error(y_test2, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test2, y_pred)
print(f"\nRMSE финальной модели: {rmse:.2f} ℃")
print(f"R²  финальной модели: {r2:.3f}")

# 9. Визуализация feature importances (все признаки)
plt.figure(figsize=(10, 6))
plt.barh(feat_imp_df['feature'], feat_imp_df['importance'])
plt.axvline(threshold, color='red', linestyle='--', label=f'Порог = {threshold}')
plt.gca().invert_yaxis()
plt.xlabel('Importance')
plt.title('CatBoost Feature Importances (initial model)')
plt.legend()
plt.tight_layout()
plt.show()
