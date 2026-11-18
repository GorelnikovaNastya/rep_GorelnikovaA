from src.realty_utils import train_model

result = train_model()

print("Обучение завершено!")
print("Строк для обучения:", result["train_rows"])
print("Строк для проверки:", result["test_rows"])
print("Средняя абсолютная ошибка:", result["mae"])
print("Модель сохранена в realty_model.pkl")
