import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import ace_tools_open

# 1. Загрузка набора данных
df = pd.read_csv('ВСТАВТЕ ПУТЬ ДО exel-ФАЙЛА')

# Вывод списка признаков
columns_df = pd.DataFrame({'Признаки': df.columns})
ace_tools_open.display_dataframe_to_user("Признаки набора данных", columns_df)

# 2. Реализация собственной линейной регрессии (нормальное уравнение)
class MyLinearRegression:
    def fit(self, X, y):
        # Добавляем столбец единиц (свободный член)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Решение через псевдо-обратную матрицу
        self.theta_ = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta_

# Для примера используем только числовые признаки
numeric_features = ['Year', 'Kilometers', 'CC', 'Seating Capacity']
X = df[numeric_features].values
y = df['Price'].values

# 3. Деление на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.255, random_state=42
)

# Обучение пользовательской модели
model = MyLinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Оценка качества
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Результаты пользовательской линейной регрессии на тестовой выборке:")
print(f"  R² (коэффициент детерминации): {r2:.3f}")
print(f"  MAE (средняя абсолютная ошибка): {mae:,.0f} у.е.")

# Вывод нескольких примеров предсказаний
sample = pd.DataFrame({
    'Фактическая цена': y_test[:10],
    'Предсказанная цена': y_pred[:10].round(0)
})
ace_tools_open.display_dataframe_to_user("Примеры предсказаний (первые 10 из тестовой выборки)", sample)