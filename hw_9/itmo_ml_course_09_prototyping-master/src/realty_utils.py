from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# Корневая папка проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# Исходный CSV и файл будущей модели
DATA_PATH = BASE_DIR / "data" / "realty_data.csv"
MODEL_PATH = BASE_DIR / "realty_model.pkl"

# По каким столбцам предсказываем
FEATURES = ["total_square", "rooms"]

# Какой столбец предсказываем
TARGET = "price"


def prepare_data():
    #"""Читаем и очищаем данные."""

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", )

    required_columns = FEATURES + [TARGET]

    missing_columns = []
    for column in required_columns:
        if column not in df.columns:
            missing_columns.append(column)

    if missing_columns:
            raise ValueError(
            f"Не найдены столбцы: {missing_columns}. "
            f"В файле есть: {df.columns.tolist()}"
            )

    original_rows = len(df)

    # Оставляем только нужные столбцы
    df = df[required_columns].copy()

    # Преобразуем значения в числа
    for column in required_columns:
        df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
        )

    # Удаляем строки с некорректными числами
    df = df.replace(
    [np.inf, -np.inf],
    np.nan,
    )
    df = df.dropna()

    # Удаляем невозможные значения
    df = df[
    (df["total_square"] > 0)
    & (df["rooms"] >= 0)
    & (df["price"] > 0)
    ]

    # Убираем полностью одинаковые строки
    df = df.drop_duplicates()

    print(f"Строк в исходном файле: {original_rows}")
    print(f"Строк после очистки: {len(df)}")

    if len(df) < 10:
            raise ValueError(
            "После очистки осталось меньше 10 строк. "
            "Нужно проверить формат CSV и числовых значений."
            )

    return df


def train_model():
    """Обучаем, проверяем и сохраняем модель."""

    df = prepare_data()

    X = df[FEATURES]
    y = df[TARGET]

    # 80% данных — обучение, 20% — проверка
    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    )

    model = LinearRegression()

    # Обучение модели
    model.fit(X_train, y_train)

    # Прогноз для проверочной выборки
    predictions = model.predict(X_test)

    # Средняя абсолютная ошибка
    mae = mean_absolute_error(
    y_test,
    predictions,
    )

    result = {
    "model": model,
    "features": FEATURES,
    "mae": float(mae),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "area_min": float(X_train["total_square"].min()),
    "area_max": float(X_train["total_square"].max()),
    "rooms_min": float(X_train["rooms"].min()),
    "rooms_max": float(X_train["rooms"].max()),
    }

    with open(MODEL_PATH, "wb") as file:
            pickle.dump(result, file)

    return result


def read_model():
    """Загружаем сохранённую модель."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
                    "Модель ещё не создана. "
                    "Сначала запустите train_realty.py."
                    )

    with open(MODEL_PATH, "rb") as file:
                    result = pickle.load(file)

    return result