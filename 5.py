import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, log_loss, roc_auc_score
)
import ace_tools_open   # или ace_tools, если вы используете ChatGPT-UI

# 1. Загрузка датасета
df = pd.read_csv("ВСТАВТЕ ПУТЬ ДО exel-ФАЙЛА")
ace_tools_open.display_dataframe_to_user(
    "Список признаков",
    pd.DataFrame({"Признак": df.columns})
)

# 2. Выбор целевой переменной
target_col = "room_type"
df = df.dropna(subset=[target_col])  # удаляем строки без метки
y = df[target_col].values
X = df.drop(columns=[target_col])

# 3. Предобработка признаков
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Кодируем метки классов
le = LabelEncoder()
y_enc = le.fit_transform(y)

# 4. Определяем модели
models = {
    "SVM": Pipeline([
        ("prep", preprocessor),
        ("clf",  SVC(probability=True, random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ("prep", preprocessor),
        ("clf",  DecisionTreeClassifier(random_state=42))
    ]),
    "kNN": Pipeline([
        ("prep", preprocessor),
        ("clf",  KNeighborsClassifier(n_neighbors=5))
    ]),
    "SGD (логистическая регрессия)": Pipeline([
        ("prep", preprocessor),
        ("clf",  SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3,
                               random_state=42))
    ])
}

# 5. Разделение выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# 6. Обучение и оценка моделей
results = []
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted")
    f1   = f1_score(y_test, y_pred, average="weighted")

    # Log-Loss и ROC-AUC
    ll, auc = np.nan, np.nan
    if hasattr(pipe[-1], "predict_proba"):
        proba = pipe.predict_proba(X_test)
        ll = log_loss(y_test, proba)
        # бинарный или мультиклассовый AUC
        if proba.shape[1] == 2:
            auc = roc_auc_score(y_test, proba[:, 1])
        else:
            auc = roc_auc_score(
                y_test, proba,
                multi_class="ovr", average="weighted"
            )
    elif hasattr(pipe[-1], "decision_function"):
        scores = pipe.decision_function(X_test)
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, scores)
        else:
            auc = roc_auc_score(
                y_test, scores,
                multi_class="ovr", average="weighted"
            )

    results.append({
        "Модель":   name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall":    rec,
        "F1":        f1,
        "Log-Loss":  ll,
        "ROC-AUC":   auc
    })

metrics_df = pd.DataFrame(results).set_index("Модель").round(3)
ace_tools_open.display_dataframe_to_user("Метрики моделей", metrics_df)
