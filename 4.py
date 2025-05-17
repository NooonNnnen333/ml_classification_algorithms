import pandas as pd
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import ace_tools_open
from collections import Counter, defaultdict

# 1. Загрузка датасета
df = pd.read_csv('ВСТАВЬТЕ ПУТЬ ДО exel-ФАЙЛА')

# Формируем список признаков (последний столбец считается целевой переменной)
feature_names = [f'Признак_{i + 1}' for i in range(df.shape[1] - 1)]
features_df = pd.DataFrame({'Признак': feature_names})
ace_tools_open.display_dataframe_to_user("Список признаков", features_df)

# Подготовка данных
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Деление на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# 2. Реализация собственного KNN
class MyKNNClassifier:
    def __init__(self, n_neighbors=5, metric='euclidean', weights='uniform'):
        self.k = n_neighbors
        self.metric = metric
        self.weights = weights

    def _distance(self, X1, X2):
        if self.metric == 'euclidean':
            return np.linalg.norm(X1 - X2, axis=1)
        elif self.metric == 'manhattan':
            return np.sum(np.abs(X1 - X2), axis=1)
        elif self.metric == 'chebyshev':
            return np.max(np.abs(X1 - X2), axis=1)
        else:
            raise ValueError('Неизвестная метрика')

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)
        return self

    def _vote(self, neighbor_labels, neighbor_dists):
        if self.weights == 'uniform':
            counts = Counter(neighbor_labels)
            return max(counts.items(), key=lambda x: (x[1], -x[0]))[0]
        elif self.weights == 'distance':
            weights_sum = defaultdict(float)
            for lbl, dist in zip(neighbor_labels, neighbor_dists):
                weights_sum[lbl] += 1.0 / (dist + 1e-9)
            return max(weights_sum.items(), key=lambda x: (x[1], -x[0]))[0]
        else:
            raise ValueError('Неизвестная схема весов')

    def _proba(self, neighbor_labels, neighbor_dists):
        weights = np.ones_like(neighbor_dists)
        if self.weights == 'distance':
            weights = 1.0 / (neighbor_dists + 1e-9)
        probs = np.zeros(len(self.classes_))
        for lbl, w in zip(neighbor_labels, weights):
            idx = np.where(self.classes_ == lbl)[0][0]
            probs[idx] += w
        return probs / probs.sum()

    def predict(self, X):
        preds = []
        for x in X:
            dists = self._distance(self.X_train, x)
            idx = np.argsort(dists)[:self.k]
            preds.append(self._vote(self.y_train[idx], dists[idx]))
        return np.array(preds)

    def predict_proba(self, X):
        probas = []
        for x in X:
            dists = self._distance(self.X_train, x)
            idx = np.argsort(dists)[:self.k]
            probas.append(self._proba(self.y_train[idx], dists[idx]))
        return np.array(probas)


# 3–4. Подбор параметров
metrics = ['euclidean', 'manhattan', 'chebyshev']
weight_options = ['uniform', 'distance']
grid_results = []

for k, metric, weight in product(range(1, 11), metrics, weight_options):
    clf = MyKNNClassifier(n_neighbors=k, metric=metric, weights=weight)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    grid_results.append({'k': k, 'метрика': metric, 'веса': weight, 'точность': acc})

results_df = pd.DataFrame(grid_results).sort_values(by='точность', ascending=False).reset_index(drop=True)
ace_tools_open.display_dataframe_to_user("Таблица подбора параметров", results_df)

# 5. Лучшая модель
best = results_df.iloc[0]
best_clf = MyKNNClassifier(int(best['k']), best['метрика'], best['веса'])
best_clf.fit(X_train, y_train)

probas = best_clf.predict_proba(X_test)
preds = best_clf.predict(X_test)
classes = best_clf.classes_

summary_df = pd.DataFrame({
    'Факт': y_test,
    'Прогноз': preds,
    **{f"P(класс {c})": probas[:, i] for i, c in enumerate(classes)}
})

ace_tools_open.display_dataframe_to_user("Вероятности классов (тест, первые 10)", summary_df.head(10))
