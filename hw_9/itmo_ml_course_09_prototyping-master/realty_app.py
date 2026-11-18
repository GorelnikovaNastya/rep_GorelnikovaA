import pandas as pd
import streamlit as st

from src.realty_utils import read_model


st.set_page_config(
page_title="Прогноз стоимости квартиры",
)

st.title("Прогноз стоимости квартиры")

st.write(
"Введите площадь квартиры и количество комнат, "
"затем нажмите кнопку расчёта."
)

st.caption(
"Стоимость выводится в тех же денежных единицах, "
"что и столбец price в исходных данных."
)

try:
    saved = read_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

model = saved["model"]

area = st.number_input(
"Общая площадь, м²",
min_value=1.0,
value=50.0,
step=0.5,
)

rooms = st.number_input(
"Количество комнат",
min_value=0,
value=2,
step=1,
)

if st.button("Рассчитать стоимость"):
    input_data = pd.DataFrame(
    {
    "total_square": [area],
    "rooms": [rooms],
    }
    )

    input_data = input_data[saved["features"]]

    outside_range = (
    area < saved["area_min"]
    or area > saved["area_max"]
    or rooms < saved["rooms_min"]
    or rooms > saved["rooms_max"]
    )

    if outside_range:
        st.warning(
        "Введённые параметры выходят за диапазон "
        "обучающих данных. Прогноз может быть ненадёжным."
        )

    prediction = float(
    model.predict(input_data)[0]
    )

    if prediction <= 0:
        st.error(
        "Получилась неположительная стоимость. "
        "Для таких параметров модель дала "
        "непригодный прогноз."
        )
    else:
        formatted_price = (
        f"{prediction:,.0f}".replace(",", " ")
        )

        st.success(
        f"Прогнозируемая стоимость: {formatted_price}"
        )

        st.write(
        f"Параметры квартиры: {area:g} м², "
        f"комнат — {rooms}."
        )

with st.expander("Информация о модели"):
    st.write("Модель: линейная регрессия")
    st.write(
    f"Строк для обучения: {saved['train_rows']}"
    )
    st.write(
    f"Строк для проверки: {saved['test_rows']}"
    )

    formatted_mae = (
    f"{saved['mae']:,.0f}".replace(",", " ")
    )

    st.write(
    f"Средняя абсолютная ошибка: {formatted_mae}"
    )

st.caption(
"Учебная оценка по площади и количеству комнат. "
"Город, район, состояние квартиры и другие "
"характеристики эта модель не учитывает."
)