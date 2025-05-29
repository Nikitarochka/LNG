import pandas as pd
from orbit.models import DLT, LGT, ETS, KTR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from scipy import stats
import os
import plotly.graph_objects as go
import streamlit as st
import base64
import io
from datetime import datetime
import optuna
from datetime import timedelta

st.set_page_config(page_title="Прогноз параметров теплообменных аппаратов СПГ завода", layout="wide")

# Заголовок страницы
st.title("Прогноз параметров теплообменных аппаратов СПГ завода")

# Боковая панель с элементами управления
st.sidebar.header("Настройки")

# Глобальные переменные для гиперпараметров модели (инициализируем значениями по умолчанию)
estimator_dlt = 'stan-mcmc'
global_trend_option_dlt = 'linear'
n_bootstrap_draws_dlt = 500
regression_penalty_dlt = 'fixed_ridge'
estimator_lgt = 'stan-mcmc'
n_bootstrap_draws_lgt = 500
regression_penalty_lgt = 'fixed_ridge'
estimator_ets = 'stan-mcmc'
n_bootstrap_draws_ets = 500
estimator_ktr = 'pyro-svi'
n_bootstrap_draws_ktr = 500
num_steps_ktr = 200

# Хранилище истории прогнозов
if 'history' not in st.session_state: 
    st.session_state.history = [] 

# Функция для добавления прогноза в историю
def add_to_history(well_name, horizon, forecast_combined, forecast_test, forecast, model_name, rmse, mae, fig_original, target_parameter, 
                   estimator_dlt, estimator_lgt, estimator_ets, estimator_ktr, global_trend_option_dlt, n_bootstrap_draws_dlt, n_bootstrap_draws_lgt,
                   n_bootstrap_draws_ets, n_bootstrap_draws_ktr, regression_penalty_dlt, regression_penalty_lgt, num_steps_ktr, fill_method):
    st.session_state.history.append({
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "well_name": well_name,
        "horizon": horizon,
        "forecast_combined": forecast_combined,
        "forecast_test": forecast_test,
        "forecast": forecast,
        "model_name": model_name,
        "rmse": rmse,
        "mae": mae,
        "fig_original": fig_original,
        "target_parameter": target_parameter,
        "estimator_dlt": estimator_dlt,
        "estimator_lgt": estimator_lgt,
        "estimator_ets": estimator_ets,
        "estimator_ktr": estimator_ktr,
        "global_trend_option_dlt": global_trend_option_dlt,
        "n_bootstrap_draws_dlt": n_bootstrap_draws_dlt,
        "n_bootstrap_draws_lgt": n_bootstrap_draws_lgt,
        "n_bootstrap_draws_ets": n_bootstrap_draws_ets,
        "n_bootstrap_draws_ktr": n_bootstrap_draws_ktr,
        "regression_penalty_dlt": regression_penalty_dlt,
        "regression_penalty_lgt": regression_penalty_lgt,
        "num_steps_ktr": num_steps_ktr,
        "fill_method": fill_method
    })
            
# Загрузка файла
uploaded_file = st.sidebar.file_uploader("Анализируемый файл", type=["xlsx"], help="Выберите файл с данными в формате xlsx.")

# Инструкции для пользователя
st.markdown("""
    ## Инструкции
    1. Загрузите файл с данными в формате xlsx.
    2. Выберите теплообменник из списка.
    3. Выберите параметр для прогнозирования.
    4. Введите количество дней прогноза.
    5. Выберите модель для прогнозирования.
    6. Нажмите кнопку "Запустить прогноз".
    7. Посмотрите результаты прогноза и график.
    8. Нажмите кнопку "Перезапустить" для запуска нового прогноза.
    9. Нажмите кнопку "Просмотреть историю прогнозов" для просмотра прошлых прогнозов.
    10. Нажмите кнопку "Анализ пропущенных данных" для анализа пропущенных значений в выбранном параметре.

    **Рекомендуется использовать модель DLT для анализа параметров теплообменник, потому что модель учитывает другие параметры скважин.**\n
    **Рекомендуется использовать интерполяцию в качестве заполнения пропусков и нулей. Медиана может привести к ошибкам при наличии выбросов.**
    """)

# Добавление состояния приложения
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.well_names = []
    st.session_state.available_parameters = []

if uploaded_file is not None:

    # Сохранение файла
    file_path = os.path.join("uploaded_files", uploaded_file.name)
    os.makedirs("uploaded_files", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Определяем названия листов
    xls = pd.ExcelFile(file_path)
    data_sheets = xls.sheet_names    
   
    # Считываем названия скважин (со второго листа, т.к. первый - информационный)
    df_wells = pd.read_excel(file_path, sheet_name=data_sheets[0], header=None)
    st.session_state.well_names = df_wells.iloc[0, 1:].tolist()

    # Считываем доступные параметры (названия листов)
    st.session_state.available_parameters = data_sheets
    st.session_state.data_loaded = True

# Вывод предупреждения при отсутствии данных
if not st.session_state.data_loaded:
    st.warning("Загрузите файл с данными в формате xlsx.")
    st.stop()

# Интерфейс
st.sidebar.header("Выбор  теплообменника")
selected_wells = st.sidebar.multiselect("Выберите теплообменник", st.session_state.well_names, help="Выберите скважину(ы) для прогнозирования.")

st.sidebar.header("Выбор параметра")
target_parameter = st.sidebar.selectbox("Параметр для прогнозирования:", st.session_state.available_parameters, help="Выберите параметр скважины для прогнозирования.")



parameter_change = st.sidebar.selectbox("Параметр, который вы хотите изменить:", st.session_state.available_parameters, help="Выберите параметр скважины для изменения.")
value_change = st.sidebar.number_input("На какую величину?", value=25, help="Укажите в абсолютном значении разницу между текущей и желаемой величиной на конец прогноза")



horizon = st.sidebar.number_input("Количество дней прогноза:", min_value=1, value=30, help="Укажите количество дней для прогноза.")

st.sidebar.header("Выбор модели")
model_options = ["DLT", "LGT", "ETS", "KTR"]
selected_model = st.sidebar.selectbox("Модель:", model_options, help="Выберите модель для прогнозирования.")

# Кнопка для описания моделей
if st.sidebar.button("Описание моделей"):
    st.sidebar.markdown("**DLT (Dynamic Linear Trend):**  Эта модель подходит для прогнозирования временных рядов с трендом, сезонностью и автокорреляцией.  Она учитывает влияние регрессоров, но требует, чтобы данные были стационарными (не имели тренда).")
    st.sidebar.markdown("**LGT (Local Linear Trend):**  Эта модель похожа на DLT, но она менее требовательна к стационарности данных.")
    st.sidebar.markdown("**ETS (Exponential Smoothing):**  Эта модель  подходит для прогнозирования временных рядов с сезонностью и автокорреляцией.  Она не учитывает регрессоры.")
    st.sidebar.markdown("**KTR (Kalman Trend Regression):**  Эта модель  подходит для прогнозирования временных рядов с трендом, сезонностью, автокорреляцией и влиянием регрессоров.  Она требует, чтобы данные были стационарными.")

# Функция для создания фрейма данных по выбранной скважине
def create_well_dataframe(well_name, data_sheets, file):
    dfs = {}
    for sheet_name in data_sheets:
        df = pd.read_excel(file, sheet_name=sheet_name, header=None)
        df.columns = df.iloc[0]
        df = df[1:]
        df.index = pd.to_datetime(df.iloc[:, 0])
        dfs[sheet_name] = df[well_name]
    df_combined = pd.DataFrame(dfs)
    df_combined['ds'] = df_combined.index
    return df_combined

# Кнопка "Сравнить модели"
if st.sidebar.button("Сравнить модели"):
    if selected_wells:
        for well_name in selected_wells:
            # Создание DataFrame
            df_well = create_well_dataframe(well_name, data_sheets, file_path)
            df_well_original = df_well.copy()  

            # Обработка пропущенных значений
            df_well.replace('', np.nan, inplace=True)
            df_well.replace(0, np.nan, inplace=True)
            df_well = df_well.interpolate(method='linear', limit_direction='both')
            df_well = df_well.dropna()

            # Разделение данных на обучающие и тестовые выборки
            train_size_train = int(len(df_well) * 0.7)
            train_size_test = int(len(df_well) * 1)
            df_train = df_well[:train_size_train]
            df_test = df_well[:train_size_test]

            # Обучение модели
            df_train = df_well[df_well[target_parameter] != 0].copy() 
            df_train.fillna(method='ffill', inplace=True)

            # Выбираем соответствующие регрессоры на основе корреляции
            correlation_matrix = df_train.corr()
            relevant_regressors = correlation_matrix[target_parameter].sort_values(ascending=False).index[1:].tolist()

            # Проверка корреляции:
            #for regressor in relevant_regressors:
                # Проверяем корреляцию с целевой переменной
                #correlation = df_train[regressor].corr(df_train[target_parameter])
                #if abs(correlation) < 0.1:  
                    #st.write(f"Низкая корреляция: {regressor} (r = {correlation:.2f}). Не учитываем его в модели.")
                    #relevant_regressors.remove(regressor)

            # Убираем 'ds' из регрессоров
            relevant_regressors = [x for x in relevant_regressors if x != 'ds']

            # Оцениваем p-value для каждого регрессора
            for regressor in relevant_regressors:
                try:
                    # Используем тест Спирмена для проверки корреляции
                    correlation, p_value = stats.spearmanr(df_train[regressor], df_train[target_parameter])
                except ValueError:
                    st.write(f"Ошибка: Невозможно рассчитать тест Спирмена для {regressor}. Все значения x одинаковы.")

            # Фильтруем регрессоры с p-value < 0.01:
            significant_regressors = [
                regressor for regressor in relevant_regressors
                if stats.spearmanr(df_train[regressor], df_train[target_parameter])[1] < 0.99
            ]
            regressor_cols = significant_regressors

            st.write(f"## Сравнение моделей для скважины {well_name}")
            comparison_results = {}

            for model_name in ["DLT", "LGT", "ETS", "KTR"]:
                if model_name == "DLT":
                    model = DLT(response_col=target_parameter, regressor_col=significant_regressors)
                elif model_name == "LGT":
                    model = LGT(response_col=target_parameter, regressor_col=significant_regressors)
                elif model_name == "ETS":
                    model = ETS(response_col=target_parameter)
                elif model_name == "KTR":
                    model = KTR(response_col=target_parameter, regressor_col=significant_regressors)
                
                model.fit(df=df_train)

                # Генерируем прогноз на тестовой выборке
                forecast_test = model.predict(df_test)

                # Оцениваем точность модели
                mse = mean_squared_error(df_test[target_parameter], forecast_test["prediction"])
                rmse = mean_squared_error(df_test[target_parameter], forecast_test["prediction"], squared=False)

                comparison_results[model_name] = {"MSE": mse, "RMSE": rmse}

            # Выводим результаты сравнения
            st.markdown("### Результаты сравнения:")
            df_comparison = pd.DataFrame(comparison_results).T
            st.dataframe(df_comparison)

            # Рекомендация по выбору модели
            best_model = df_comparison["RMSE"].idxmin()
            st.write(f"**Рекомендованная модель для скважины {well_name}:** {best_model}")
    else:
        st.warning("Выберите скважину(ы) для сравнения моделей.")

# Элементы боковой панели для выбора гиперпараметров
# Настройка модели DLT
if selected_model == "DLT":
    if 'estimator_dlt' not in st.session_state:
        st.session_state.estimator_dlt = 'stan-mcmc'
    estimator_dlt = st.sidebar.selectbox('Estimator DLT', ['stan-map', 'stan-mcmc'], index=list(['stan-map', 'stan-mcmc']).index(st.session_state.estimator_dlt))
    if 'global_trend_option_dlt' not in st.session_state:
        st.session_state.global_trend_option_dlt = 'linear'
    global_trend_option_dlt = st.sidebar.selectbox('Global Trend Option DLT', ['flat', 'linear', 'loglinear', 'logistic'], index=list(['flat', 'linear', 'loglinear', 'logistic']).index(st.session_state.global_trend_option_dlt))
    if 'n_bootstrap_draws_dlt' not in st.session_state:
        st.session_state.n_bootstrap_draws_dlt = 500
    n_bootstrap_draws_dlt = st.sidebar.slider('N Bootstrap Draws DLT', 100, 1000, st.session_state.n_bootstrap_draws_dlt)
    if 'regression_penalty_dlt' not in st.session_state:
        st.session_state.regression_penalty_dlt = 'fixed_ridge'
    regression_penalty_dlt = st.sidebar.selectbox('Regression Penalty DLT', ['fixed_ridge', 'lasso', 'auto_ridge'], index=list(['fixed_ridge', 'lasso', 'auto_ridge']).index(st.session_state.regression_penalty_dlt))

# Настройка модели LGT
if selected_model == "LGT":
    if 'estimator_lgt' not in st.session_state:
        st.session_state.estimator_lgt = 'stan-mcmc'
    estimator_lgt = st.sidebar.selectbox('Estimator LGT', ['stan-mcmc', 'pyro-svi'], index=list(['stan-mcmc', 'pyro-svi']).index(st.session_state.estimator_lgt))
    if 'n_bootstrap_draws_lgt' not in st.session_state:
        st.session_state.n_bootstrap_draws_lgt = 500
    n_bootstrap_draws_lgt = st.sidebar.slider('N Bootstrap Draws LGT', 100, 1000, st.session_state.n_bootstrap_draws_lgt)
    if 'regression_penalty_lgt' not in st.session_state:
        st.session_state.regression_penalty_lgt = 'fixed_ridge'
    regression_penalty_lgt = st.sidebar.selectbox('Regression Penalty LGT', ['fixed_ridge', 'lasso', 'auto_ridge'], index=list(['fixed_ridge', 'lasso', 'auto_ridge']).index(st.session_state.regression_penalty_lgt))

# Настройка модели ETS
if selected_model == "ETS":
    if 'estimator_ets' not in st.session_state:
        st.session_state.estimator_ets = 'stan-mcmc'
    estimator_ets = st.sidebar.selectbox('Estimator ETS', ['stan-map', 'stan-mcmc'], index=list(['stan-map', 'stan-mcmc']).index(st.session_state.estimator_ets))
    if 'n_bootstrap_draws_ets' not in st.session_state:
        st.session_state.n_bootstrap_draws_ets = 500
    n_bootstrap_draws_ets = st.sidebar.slider('N Bootstrap Draws ETS', 100, 1000, st.session_state.n_bootstrap_draws_ets)

# Настройка модели KTR
if selected_model == "KTR":
    if 'estimator_ktr' not in st.session_state:
        st.session_state.estimator_ktr = 'pyro-svi'
    estimator_ktr = st.sidebar.selectbox('Estimator KTR', ['pyro-svi'], index=list(['pyro-svi']).index(st.session_state.estimator_ktr))
    if 'n_bootstrap_draws_ktr' not in st.session_state:
        st.session_state.n_bootstrap_draws_ktr = 500
    n_bootstrap_draws_ktr = st.sidebar.slider('N Bootstrap Draws KTR', 100, 1000, st.session_state.n_bootstrap_draws_ktr)
    if 'num_steps_ktr' not in st.session_state:
        st.session_state.num_steps_ktr = 200
    num_steps_ktr = st.sidebar.slider('Num Steps KTR', 100, 500, st.session_state.num_steps_ktr)
    
# Кнопка "Дополнительная информация"
if st.sidebar.button("Дополнительная информация"):
    if selected_model == "DLT":
        st.sidebar.markdown("**Estimator DLT:**  Метод, используемый для обучения модели.  Stan-map - это более быстрый, но менее точный метод.  Stan-mcmc - это более точный, но более медленный метод.")
        st.sidebar.markdown("**Global trend option DLT:**  Тип глобального тренда в данных.  'flat' - отсутствие тренда, 'linear' - линейный тренд, 'loglinear' - логарифмический тренд, 'logistic' - логистический тренд.")
        st.sidebar.markdown("**Bootstrap draws DLT:**  Количество сэмплов, которые используются для оценки неопределенности прогноза.  Увеличьте это значение, чтобы получить более точную оценку неопределенности.")
        st.sidebar.markdown("**Regression penalty DLT:**  Тип регуляризации для регрессии.  'fixed_ridge' - фиксированная регуляризация типа L2, 'lasso' - регуляризация типа L1, 'auto_ridge' - автоматический выбор типа регуляризации. ")
    elif selected_model == "LGT":
        st.sidebar.markdown("**Estimator LGT:**  Метод, используемый для обучения модели.  Stan-mcmc - это более точный, но более медленный метод.  Pyro-svi - это более быстрый, но менее точный метод.")
        st.sidebar.markdown("**Bootstrap draws LGT:**  Количество сэмплов, которые используются для оценки неопределенности прогноза.  Увеличьте это значение, чтобы получить более точную оценку неопределенности.")
        st.sidebar.markdown("**Regression penalty LGT:**  Тип регуляризации для регрессии.  'fixed_ridge' - фиксированная регуляризация типа L2, 'lasso' - регуляризация типа L1, 'auto_ridge' - автоматический выбор типа регуляризации. ")
    elif selected_model == "ETS":
        st.sidebar.markdown("**Estimator ETS:**  Метод, используемый для обучения модели.  Stan-map - это более быстрый, но менее точный метод.  Stan-mcmc - это более точный, но более медленный метод.")
        st.sidebar.markdown("**Bootstrap draws ETS:**  Количество сэмплов, которые используются для оценки неопределенности прогноза.  Увеличьте это значение, чтобы получить более точную оценку неопределенности.")
    elif selected_model == "KTR":
        st.sidebar.markdown("**Estimator KTR:**  Метод, используемый для обучения модели.  Pyro-svi - это единственный метод у модели.")
        st.sidebar.markdown("**Bootstrap draws KTR:**  Количество сэмплов, которые используются для оценки неопределенности прогноза.  Увеличьте это значение, чтобы получить более точную оценку неопределенности.")
        st.sidebar.markdown("**Num steps KTR:**  Количество шагов, которые используются для обучения модели.  Увеличение этого значения может улучшить точность модели, но обучение будет дольше.")

# Кнопка для запуска автоматической настройки
if st.sidebar.button("Автонастройка"):
    for well_name in selected_wells:
        # Создание DataFrame
        df_well = create_well_dataframe(well_name, data_sheets, file_path)
        df_well_original = df_well.copy()  # Копия исходного DataFrame

        # Обработка пропущенных значений
        df_well.replace('', np.nan, inplace=True)
        df_well.replace(0, np.nan, inplace=True)
        df_well = df_well.interpolate(method='linear', limit_direction='both')
        df_well = df_well.dropna()

        # Разделение данных на обучающие и тестовые выборки
        train_size_train = int(len(df_well) * 0.7)
        train_size_test = int(len(df_well) * 1)
        df_train = df_well[:train_size_train]
        df_test = df_well[:train_size_test]

        #df_test = df_well[df_well[target_parameter] != 0].copy()  
        #df_test.fillna(method='ffill', inplace=True) 
        
        # Обучение модели
        df_train = df_well[df_well[target_parameter] != 0].copy()  # Обучение только на ненулевых значениях
        df_train.fillna(method='ffill', inplace=True) # Заполняем нулевые значения регрессоров при обучении 

        # Выбираем соответствующие регрессоры на основе корреляции
        correlation_matrix = df_train.corr()
        relevant_regressors = correlation_matrix[target_parameter].sort_values(ascending=False).index[1:].tolist()

        # Проверка корреляции:
        #for regressor in relevant_regressors:
            # Проверяем корреляцию с целевой переменной
            #correlation = df_train[regressor].corr(df_train[target_parameter])
            #if abs(correlation) < 0.1:  
                #st.write(f"Низкая корреляция: {regressor} (r = {correlation:.2f}). Не учитываем его в модели.")
                #relevant_regressors.remove(regressor)

        # Убираем 'ds' из регрессоров
        relevant_regressors = [x for x in relevant_regressors if x != 'ds']

        # Оцениваем p-value для каждого регрессора
        for regressor in relevant_regressors:
            try:
                # Используем тест Спирмена для проверки корреляции
                correlation, p_value = stats.spearmanr(df_train[regressor], df_train[target_parameter])
            except ValueError:
                st.write(f"Ошибка: Невозможно рассчитать тест Спирмена для {regressor}. Все значения x одинаковы.")

        # Фильтруем регрессоры с p-value < 0.01:
        significant_regressors = [
            regressor for regressor in relevant_regressors
            if stats.spearmanr(df_train[regressor], df_train[target_parameter])[1] < 0.99
        ]

    def objective(trial):
        # Определение гиперпараметров для каждой модели
        if selected_model == "DLT":  # Если выбрана модель DLT
            estimator_dlt = trial.suggest_categorical('estimator_dlt', ['stan-map', 'stan-mcmc'])  # Выбор метода оптимизации для DLT
            global_trend_option_dlt = trial.suggest_categorical('global_trend_option_dlt', ['flat', 'linear', 'loglinear', 'logistic'])  # Выбор типа тренда для DLT
            n_bootstrap_draws_dlt = trial.suggest_int('n_bootstrap_draws_dlt', 100, 1000)  # Количество бутстреп-выборок для DLT
            regression_penalty_dlt = trial.suggest_categorical('regression_penalty_dlt', ['fixed_ridge', 'lasso', 'auto_ridge'])  # Регуляризация для DLT
            model = DLT(response_col=target_parameter, regressor_col=significant_regressors,  # Создание объекта модели DLT
                        estimator=estimator_dlt, seasonality=12,  # Задание параметров модели
                        global_trend_option=global_trend_option_dlt, n_bootstrap_draws=n_bootstrap_draws_dlt,
                        regression_penalty=regression_penalty_dlt)
        elif selected_model == "LGT":  # Аналогично для модели LGT
            estimator_lgt = trial.suggest_categorical('estimator_lgt', ['stan-mcmc', 'pyro-svi'])
            n_bootstrap_draws_lgt = trial.suggest_int('n_bootstrap_draws_lgt', 100, 1000)
            regression_penalty_lgt = trial.suggest_categorical('regression_penalty_lgt', ['fixed_ridge', 'lasso', 'auto_ridge'])
            model = LGT(response_col=target_parameter, regressor_col=significant_regressors,
                        estimator=estimator_lgt, seasonality=12,  # Задание параметров модели
                        regression_penalty=regression_penalty_lgt)
        elif selected_model == "ETS":  # Аналогично для модели ETS
            estimator_ets = trial.suggest_categorical('estimator_ets', ['stan-mcmc', 'pyro-svi'])
            n_bootstrap_draws_ets = trial.suggest_int('n_bootstrap_draws_ets', 100, 1000)
            model = ETS(response_col=target_parameter,
                        estimator=estimator_ets, seasonality=12,  # Задание параметров модели
                        n_bootstrap_draws=n_bootstrap_draws_ets)
        elif selected_model == "KTR":  # Аналогично для модели KTR
            estimator_ktr = trial.suggest_categorical('estimator_ktr', ['pyro-svi'])
            n_bootstrap_draws_ktr = trial.suggest_int('n_bootstrap_draws_ktr', 100, 1000)
            num_steps_ktr = trial.suggest_int('num_steps_ktr', 100, 500)
            model = KTR(response_col=target_parameter, regressor_col=significant_regressors,
                        estimator=estimator_ktr, seasonality=12,  # Задание параметров модели
                        n_bootstrap_draws=n_bootstrap_draws_ktr, num_steps=num_steps_ktr)

        # Обучение модели
        try:
            model.fit(df=df_train)  # Обучение модели на тренировочных данных
        except ValueError:
            st.write(f"Ошибка: Невозможно обучить модель {selected_model} для скважины {well_name}. Проверьте данные.")  # Вывод сообщения об ошибке, если обучение не удалось
            return float('inf')  # Возвращаем бесконечность, чтобы Optuna не выбрал этот набор гиперпараметров
        
        # Получение прогноза
        forecast_test = model.predict(df_test)  # Получение прогнозов на тестовых данных
        
        # Оценка модели                     
        try:
            rmse = mean_squared_error(df_test[target_parameter].astype(float), forecast_test["prediction"], squared=False)  # Расчет RMSE
        except ValueError:
            st.write(f"Ошибка: Невозможно рассчитать RMSE для модели {selected_model}. Проверьте данные.") 
            return float('inf') 
        return rmse  # Возвращаем значение RMSE

    # Запуск оптимизации Optuna
    study = optuna.create_study(direction="minimize")  # Создание объекта Optuna для поиска минимального значения RMSE
    study.optimize(objective, n_trials=10)  # Запуск оптимизации, n_trials - количество испытаний 

    # Вывод результатов
    best_params = study.best_params  # Получение лучших найденных гиперпараметров
    st.markdown(f"\n**Наиболее оптимальные гиперпараметры: {best_params}.**") # Вывод оптимальных гиперпараметров на экран
    st.write(f"\n**Пожалуйста, укажите их в панели настройки.**")
    
    if selected_model == "DLT":
        st.session_state.estimator_dlt = best_params.get('estimator_dlt', 'stan-mcmc')
        st.session_state.global_trend_option_dlt = best_params.get('global_trend_option_dlt', 'linear')
        st.session_state.n_bootstrap_draws_dlt = best_params.get('n_bootstrap_draws_dlt', 500)
        st.session_state.regression_penalty_dlt = best_params.get('regression_penalty_dlt', 'fixed_ridge')
    elif selected_model == "LGT":
        st.session_state.estimator_lgt = best_params.get('estimator_lgt', 'stan-mcmc')
        st.session_state.n_bootstrap_draws_lgt = best_params.get('n_bootstrap_draws_lgt', 500)
        st.session_state.regression_penalty_lgt = best_params.get('regression_penalty_lgt', 'fixed_ridge')
    elif selected_model == "ETS":
        st.session_state.estimator_ets = best_params.get('estimator_ets', 'stan-mcmc')
        st.session_state.n_bootstrap_draws_ets = best_params.get('n_bootstrap_draws_ets', 500)
    elif selected_model == "KTR":
        st.session_state.estimator_ktr = best_params.get('estimator_ktr', 'pyro-svi')
        st.session_state.n_bootstrap_draws_ktr = best_params.get('n_bootstrap_draws_ktr', 500)
        st.session_state.num_steps_ktr = best_params.get('num_steps_ktr', 200)

st.sidebar.header("Настройки графика")
graph_color = st.sidebar.color_picker("Прогноз", value="#FF0000", help="Выберите цвет линии прогноза.")
graph_color_test = st.sidebar.color_picker("Тестовый прогноз", value="#00FF00", help="Выберите цвет линии прогноза на тестовых данных.")
graph_color_actual = st.sidebar.color_picker("Фактические данные", value="#000000", help="Выберите цвет линии фактических данных.")
graph_line_width = st.sidebar.slider("Толщина линии", min_value=1, max_value=5, value=2, help="Установите толщину линий на графике.")
graph_font_size = st.sidebar.slider("Размер шрифта", min_value=10, max_value=20, value=12, help="Установите размер шрифта для текста на графике.")
show_regressors = st.sidebar.checkbox("Отобразить релевантные регрессоры на графике прогноза", value=True)
background_color = st.sidebar.color_picker("Цвет фона", value="#FFFFFF", help="Выберите цвет фона для графика.")

st.sidebar.header("Выбор способа заполнения")

# Заполнение пропущенных значений
fill_method = st.sidebar.radio(
    "Способ:",
    ("Интерполяция", "Медиана ненулевых значений"),
    help="Выберете способ заполнения пропусков и нулей у регрессоров."
)

# Кнопка "Информация о методах заполнения"
if st.sidebar.button("Информация о методах заполнения"):
    st.sidebar.markdown("**Интерполяция:**  Этот метод позволяет заполнить пропущенные значения плавно,  сохраняя общий тренд данных.  Он лучше всего подходит для данных, которые имеют плавное изменение.")
    st.sidebar.markdown("**Медиана ненулевых значений:**  Этот метод  заполняет пропущенные значения медианой ненулевых значений.  Он подходит для данных, которые могут иметь выбросы.")

st.markdown("## Результаты прогнозирования")

# Функция для загрузки всех данных с листов
def load_all_data(file_path, target_parameter, well_names):
    
    # Создаем список для хранения всех DataFrames
    dfs_list = []

    for sheet_name in data_sheets:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        df.columns = df.iloc[0]
        df = df[1:]
        df.index = pd.to_datetime(df.iloc[:, 0])
        dfs_list.append(df[well_names])

    # Объединяем DataFrames по столбцам
    df_combined = pd.concat(dfs_list, axis=1)
    df_combined['ds'] = df_combined.index
    return df_combined.reset_index()

# Функция для анализа пропущенных данных
def analyze_missing_data(df_well, target_parameter):
    st.markdown("## Анализ пропущенных данных")

    for well_name in selected_wells:
        st.markdown(f"### Скважина {well_name}")
        df = df_well[df_well.columns.drop('ds')]
        for col in df.columns:
            total_values = len(df[col])
            null_values = df[col].isnull().sum()
            zero_values = (df[col] == 0).sum()
            missing_percentage = (null_values / total_values) * 100
            zero_percentage = (zero_values / total_values) * 100
            
            st.markdown(f"**Параметр: {col}**")
            st.markdown(f"   - Общее количество значений: {total_values}")
            st.markdown(f"   - Количество пропущенных значений: {null_values} ({missing_percentage:.2f}%)")
            st.markdown(f"   - Количество нулевых значений: {zero_values} ({zero_percentage:.2f}%)")

# Кнопка запуска прогноза
if st.button("Запустить прогноз"):
    # Запуск прогнозирования для нескольких скважин
    for well_name in selected_wells:
        # Создание DataFrame
        df_well = create_well_dataframe(well_name, data_sheets, file_path)
        df_well_original = df_well.copy() 

        # Обработка пропущенных значений
        df_well.replace('', np.nan, inplace=True)
        df_well.replace(0, np.nan, inplace=True)
        #df_well = df_well.interpolate(method='linear', limit_direction='both')

        # Выбор метода заполнения пропущенных значений
        if fill_method == "Интерполяция":
            df_well = df_well.interpolate(method='linear', limit_direction='both')
        elif fill_method == "Медиана ненулевых значений":
            for col in df_well.columns:
                df_well[col] = df_well[col].fillna(df_well[col].dropna().median())
                    
        df_well = df_well.dropna()

        # Разделение данных на обучающие и тестовые выборки
        train_size_train = int(len(df_well) * 0.7)
        train_size_test = int(len(df_well) * 1)
        df_train = df_well[:train_size_train]
        df_test = df_well[:train_size_test]

        # Обучение модели
        df_train = df_well[df_well[target_parameter] != 0].copy() 
        df_train.fillna(method='ffill', inplace=True) 

        # Выбираем соответствующие регрессоры на основе корреляции
        correlation_matrix = df_train.corr() 
        relevant_regressors = correlation_matrix[target_parameter].sort_values(ascending=False).index[1:].tolist()

        # Убираем 'ds' из регрессоров
        relevant_regressors = [x for x in relevant_regressors if x != 'ds']
        
        # Проверка корреляции:
        #for regressor in relevant_regressors:
            # Проверяем корреляцию с целевой переменной
            #correlation = df_train[regressor].corr(df_train[target_parameter])
            #if abs(correlation) < 0.1:
                #st.write(f"Низкая корреляция: {regressor} (r = {correlation:.2f}). Не учитываем его в модели.")
                #relevant_regressors.remove(regressor)

        # Оцениваем p-value для каждого регрессора
        for regressor in relevant_regressors:
            try:
                # Используем тест Спирмена для проверки корреляции
                correlation, p_value = stats.spearmanr(df_train[regressor], df_train[target_parameter])
                st.write(f"Регрессор: {regressor}, p-value (Спирмена): {p_value:.3f}")
            except ValueError:
                st.write(f"Ошибка: Невозможно рассчитать тест Спирмена для {regressor}. Все значения x одинаковы.")

        # Фильтруем регрессоры с p-value < 0.01:
        significant_regressors = [
            regressor for regressor in relevant_regressors
            if stats.spearmanr(df_train[regressor], df_train[target_parameter])[1] < 0.99
        ]
        regressor_cols = significant_regressors
        st.write(f'Регрессоры, которые образуют зависимость с {target_parameter}: {regressor_cols}')
            
        # Выбираем модель прогнозирования
        if selected_model == "DLT":
            model = DLT(response_col=target_parameter, regressor_col=significant_regressors, estimator=estimator_dlt, regression_penalty = regression_penalty_dlt, seasonality=12, global_trend_option=global_trend_option_dlt, n_bootstrap_draws=n_bootstrap_draws_dlt)
        elif selected_model == "LGT":
            model = LGT(response_col=target_parameter, regressor_col=significant_regressors, estimator=estimator_lgt, regression_penalty = regression_penalty_lgt, seasonality=12, n_bootstrap_draws=n_bootstrap_draws_lgt)
        elif selected_model == "ETS":
            model = ETS(response_col=target_parameter, estimator=estimator_ets, seasonality=12, n_bootstrap_draws=n_bootstrap_draws_ets)
        elif selected_model == "KTR":
            model = KTR(response_col=target_parameter, regressor_col=significant_regressors, estimator=estimator_ktr, seasonality=12, n_bootstrap_draws=n_bootstrap_draws_ktr, num_steps=num_steps_ktr)
        
        model.fit(df=df_train)
            
        # Генерируем прогноз на тестовой выборке
        forecast_test = model.predict(df_test)

        # Оцениваем точность модели
        rmse = mean_squared_error(df_test[target_parameter], forecast_test["prediction"], squared=False)
        mae = mean_absolute_error(df_test[target_parameter], forecast_test["prediction"])
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"MAE: {mae:.2f}")

        # Проверка на обводненность
        is_obvodnennost = target_parameter.lower().startswith('обводненность')

        # Генерируем прогноз на будущие даты
        last_date = df_well.index[-1]
        future_dates = pd.date_range(last_date + pd.DateOffset(days=1), periods=horizon)

        # Добавляем регрессоры в df_future
        df_future = pd.DataFrame({'ds': future_dates})

        # Добавляем регрессоры из прошлого
        for col in significant_regressors:
            df_future[col] = df_well[col].iloc[-1]

        # Если выбранный параметр для изменения находится в регрессорах
        #if parameter_change in df_future.columns:
            # Линейное изменение параметра на протяжении всего горизонта прогноза
           # df_future[parameter_change] = df_well[parameter_change].iloc[-1] + np.linspace(0, value_change, horizon)

        # Добавляем измененный параметр как регрессор
        if parameter_change:
            last_date = df_well.index[-1]
            forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=horizon)
            changed_values = np.linspace(df_well[parameter_change].iloc[-1],
                                         df_well[parameter_change].iloc[-1] + value_change, horizon)
            df_future[parameter_change] = changed_values

        # Генерируем прогноз
        forecast = model.predict(df_future)

        # Если параметр "Обводненность", ограничиваем его значение в прогнозе до 100
        if is_obvodnennost:
            forecast['prediction'] = forecast['prediction'].clip(upper=100)

        # Объединяем результаты прогноза
        forecast_combined = pd.concat([forecast_test, forecast])
        forecast_combined = forecast_combined.reset_index()
        forecast_combined = forecast_combined.rename(columns={'ds': 'Дата'})

        # Добавляем показ регрессоров на графике
        if show_regressors:
            for col in significant_regressors:
                forecast_combined[col] = df_well[col].fillna(method='ffill')

        # Создаём информативные графики
        fig_original = go.Figure()
        fig_original.add_trace(go.Scatter(x=df_well_original.index, y=df_well_original[target_parameter], mode='lines', name='Реальные данные (Исходные)', line=dict(color=graph_color_actual, width=graph_line_width)))
        fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['prediction'], mode='lines', name='Прогноз на реальных данных', line=dict(color=graph_color_test, width=graph_line_width)))
        # Добавление prediction_5 и prediction_95 и закрашиваем область
        # fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['prediction_5'], mode='lines', line=dict(color='rgba(255, 255, 255, 0)')))
        fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['prediction_95'], mode='lines',
                                          line=dict(color='rgba(255, 255, 255, 0)')))
        fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['prediction_5'], fill='tonexty',
                                          name='Доверительный интервал', fillcolor='rgba(255, 255, 208, 0.7)',
                                          line=dict(color='rgba(255, 255, 255, 0)')))

        fig_original.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['prediction'], mode='lines', name='Прогноз в будущее',
                       line=dict(color=graph_color, width=1)))
        # Добавление prediction_5 и prediction_95 и закрашиваем область
        # fig_original.add_trace(go.Scatter(x=forecast['ds'], y=forecast['prediction_5'], mode='lines', line=dict(color='rgba(255, 255, 255, 0)')))
        fig_original.add_trace(go.Scatter(x=forecast['ds'], y=forecast['prediction_95'], mode='lines',
                                          line=dict(color='rgba(255, 255, 255, 0)')))
        fig_original.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['prediction_5'], fill='tonexty', name='Доверительный интервал',
                       fillcolor='rgba(205, 51, 51, 0.7)', line=dict(color='rgba(255, 255, 255, 0)')))

        
        if parameter_change:
            last_date = df_well.index[-1]
            forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=horizon)
            changed_values = np.linspace(df_well[parameter_change].iloc[-1], df_well[parameter_change].iloc[-1] + value_change, horizon)

            df_changed = pd.DataFrame({
                'ds': forecast_dates,
                parameter_change: changed_values
            })

            fig_original.add_trace(go.Scatter(
                x=df_changed['ds'],
                y=df_changed[parameter_change],
                mode='lines',
                name=f'Изменённый {parameter_change}',
                line=dict(color='red', width=graph_line_width, dash='dash')
            ))

        if show_regressors:
            for i, col in enumerate(significant_regressors):
                fig_original.add_trace(go.Scatter(x=df_well.index, y=df_well[col], mode='lines', name=col, line=dict(width=3, dash='dash'), yaxis="y2"))

        
        # Добавляем изменённый параметр в будущее
        #fig_original.add_trace(go.Scatter(
        #    x=df_future['ds'],
         #   y=df_future[parameter_change],
        #    mode='lines',
         #   name=f'Изменение {parameter_change} в будущем',
         #   line=dict(color='purple', width=2, dash='dot')
       # ))

        # Настройки осей и оформления
        fig_original.update_layout(
            title=f"Прогноз {target_parameter} для теплообменника {well_name} (Модель: {selected_model}. Способ заполнения: {fill_method}.)",
            xaxis_title="Дата",
            yaxis_title=target_parameter,
            yaxis2=dict(
                title="Регрессоры",
                overlaying="y",
                side="right"
            ),
            height=600,
            width=1000,
            legend=dict(x=0, y=1.0),
            plot_bgcolor=background_color,
            hovermode='x',
            hoverdistance=100,
            spikedistance=-1,
            font=dict(size=graph_font_size),
            showlegend=True
        )

        # Если целевой параметр — обводненность, ограничиваем ось y
        if is_obvodnennost:
            fig_original.update_yaxes(range=[0, 100])

        # Отключаем шаблон hover и добавляем свои подписи
        fig_original.update_traces(hovertemplate=None)
        fig_original.update_traces(hovertemplate="<b>Дата:</b> %{x}<br><b>Значение:</b> %{y}<br>")

        # Добавляем график в историю прогнозов и выводим его на экран
        add_to_history(well_name, horizon, forecast_combined, forecast_test, forecast, selected_model, rmse, mae, fig_original, target_parameter, 
                       estimator_dlt, estimator_lgt, estimator_ets, estimator_ktr, global_trend_option_dlt, n_bootstrap_draws_dlt, n_bootstrap_draws_lgt,
                       n_bootstrap_draws_ets, n_bootstrap_draws_ktr, regression_penalty_dlt, regression_penalty_lgt, num_steps_ktr, fill_method)

        # Выводим график
        st.plotly_chart(fig_original)

        # Таблица для прогноза в будущее
        st.markdown("## Прогноз:")
        st.dataframe(forecast_combined)

# Кнопка перезапуска
if st.button("Перезапустить"):
    st.rerun()

# Просмотр истории прогнозов
if st.button("Просмотреть историю прогнозов"):
    if st.session_state.history:
        st.markdown("## История прогнозов")
        for item in st.session_state.history:
            st.markdown(f"**Дата и время:** {item['datetime']}")
            st.markdown(f"**Скважина:** {item['well_name']}")
            st.markdown(f"**Горизонт прогноза:** {item['horizon']} дней")
            st.markdown(f"**Модель:** {item['model_name']}")
            if item['model_name'] == "DLT":
                st.markdown(f"**Estimator:** {item['estimator_dlt']}")
                st.markdown(f"**Global_trend_option:** {item['global_trend_option_dlt']}")
                st.markdown(f"**n bootstrap draws:** {item['n_bootstrap_draws_dlt']}")
                st.markdown(f"**Regression penalty:** {item['regression_penalty_dlt']}")
            if item['model_name'] == "LGT":
                st.markdown(f"**Estimator:** {item['estimator_lgt']}")
                st.markdown(f"**n bootstrap draws:** {item['n_bootstrap_draws_lgt']}")
                st.markdown(f"**Regression penalty:** {item['regression_penalty_lgt']}")
            if item['model_name'] == "ETS":
                st.markdown(f"**Estimator:** {item['estimator_ets']}")
                st.markdown(f"**n bootstrap draws:** {item['n_bootstrap_draws_ets']}")
            if item['model_name'] == "KTR":
                st.markdown(f"**Estimator:** {item['estimator_ktr']}")
                st.markdown(f"**n bootstrap draws:** {item['n_bootstrap_draws_ktr']}")
                st.markdown(f"**Num steps:** {item['num_steps_ktr']}")
            st.markdown(f"**Способ заполнения пропусков:** {item['fill_method']}") 
            st.markdown(f"**RMSE:** {item['rmse']:.2f}")
            st.markdown(f"**MAE:** {item['mae']:.2f}")
            st.markdown(f"**Прогнозируемый параметр:** {item['target_parameter']}")
            st.plotly_chart(item['fig_original'])
            st.dataframe(item['forecast_combined'])
    else:
        st.markdown("История прогнозов пуста.")

# Кнопка для анализа пропущенных данных
if st.button("Анализ пропущенных данных"):
    if selected_wells:
        df_well = create_well_dataframe(selected_wells[0], data_sheets, file_path)
        analyze_missing_data(df_well, target_parameter)
    else:
        st.warning("Выберите скважину(ы) для анализа пропущенных данных.")

# Просмотр исходных данных
if st.button("Просмотреть исходные данные"):

    if uploaded_file is not None:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names

        for sheet_name in sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
            st.markdown(f"## Исходные данные - Лист: {sheet_name}")
            st.dataframe(df)

# Просмотр данных по выбранным скважинам
if st.button("Посмотреть данные по выбранным теплообменникам"):
    if uploaded_file is not None:
        st.markdown("## Данные по выбранным теплообменникам")
        if selected_wells:
            for well_name in selected_wells:
                df_well = create_well_dataframe(well_name, data_sheets, file_path)
                st.markdown(f"### Теплообменник {well_name}")
                st.dataframe(df_well)

                # Обновляем структуру данных для графика с точками, соединенными линиями
                df_well_plot = df_well.rename(columns={'ds': 'Даты'})  

                # Создаем график с точками, соединенными линиями
                fig_line = go.Figure()
                for column_name in df_well_plot.columns[1:]:
                    fig_line.add_trace(go.Scatter(x=df_well_plot['Даты'], y=df_well_plot[column_name], mode='lines', name=column_name))

                fig_line.update_layout(
                    hovermode='x unified',  
                    yaxis_title='Параметры теплообменника',  
                )

                st.plotly_chart(fig_line) 

                # Создаем график с распределением данных точками
                fig_scatter = go.Figure()
                for column_name in df_well_plot.columns:
                    fig_scatter.add_trace(go.Scatter(x=df_well_plot['Даты'], y=df_well_plot[column_name], mode='markers', name=column_name))

                fig_scatter.update_layout(
                    hovermode='x unified', 
                    yaxis_title='Параметры теплообменника',  
                )

                st.plotly_chart(fig_scatter)  

        else:
            st.markdown("Выберите теплообменника для просмотра данных.")
    else:
        st.markdown("Загрузите файл с данными.")
