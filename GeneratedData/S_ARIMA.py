import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import mean_absolute_error

# Функция для генерации тестовых данных с настраиваемыми параметрами
def generate_test_data(start_date='2023-01-01', periods=1000, freq='D', trend_slope=0.1,
                       seasonality_period=7, seasonality_amplitude=5, noise_scale=1, trend_type="linear"):
    t = np.arange(periods)
    series = np.zeros(periods)

    if trend_type == "linear":
        series += trend_slope * t
    elif trend_type == "exp":
        series += np.exp(trend_slope * t / periods) - 1

    seasonality = seasonality_amplitude * np.sin(2 * np.pi * t / seasonality_period)
    series += seasonality  # Добавляем сезонность, а не noise

    noise = np.random.normal(0, noise_scale, periods)
    series += noise  # Добавляем шум

    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    return pd.Series(series, index=dates)
# Класс для работы с ARIMA/SARIMA
class EnhancedARIMAForecaster:
    def __init__(self, data, train_size=0.8):
        self.data = data
        self.train_size = train_size
        self.train, self.test = self._split_data()
        self.model = None
        self.fitted_model = None
        self.predictions = None
        self.best_params = None
        self.best_seasonal_params = None

    def _split_data(self):
        split_point = int(len(self.data) * self.train_size)
        return self.data[:split_point], self.data[split_point:]

    def check_stationarity(self):
        result = adfuller(self.data)
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])
        print('Critical Values:', result[4])
        if result[1] <= 0.05:
            print("Ряд стационарен (p-value <= 0.05)")
        else:
            print("Ряд нестационарен (p-value > 0.05). Используем d=1 или SARIMA.")

    def grid_search_params(self, p_range, d_range, q_range, seasonal=False, P_range=None, D_range=None, Q_range=None, m=None):
        best_score = float('inf')
        best_params = None
        best_seasonal_params = None

        if seasonal and (P_range is None or D_range is None or Q_range is None or m is None):
            raise ValueError("Для SARIMA нужно указать P_range, D_range, Q_range и m.")

        for p, d, q in itertools.product(p_range, d_range, q_range):
            if seasonal:
                for P, D, Q in itertools.product(P_range, D_range, Q_range):
                    try:
                        model = SARIMAX(self.train, order=(p, d, q), seasonal_order=(P, D, Q, m))
                        results = model.fit(disp=False)
                        aic = results.aic
                        if aic < best_score:
                            best_score = aic
                            best_params = (p, d, q)
                            best_seasonal_params = (P, D, Q, m)
                    except:
                        continue
            else:
                try:
                    model = ARIMA(self.train, order=(p, d, q))
                    results = model.fit()
                    aic = results.aic
                    if aic < best_score:
                        best_score = aic
                        best_params = (p, d, q)
                except:
                    continue

        self.best_params = best_params
        self.best_seasonal_params = best_seasonal_params if seasonal else None
        print(f"Лучшие параметры (p,d,q): {best_params}")
        if seasonal:
            print(f"Лучшие сезонные параметры (P,D,Q,m): {best_seasonal_params}")
        print(f"AIC: {best_score}")
        return best_params, best_seasonal_params

    def fit(self, order=None, seasonal_order=None):
        if order is None and self.best_params is not None:
            order = self.best_params
            seasonal_order = self.best_seasonal_params
        elif order is None:
            raise ValueError("Укажите параметры модели или выполните grid_search_params.")

        if seasonal_order is not None:
            self.model = SARIMAX(self.train, order=order, seasonal_order=seasonal_order)
        else:
            self.model = ARIMA(self.train, order=order)
        self.fitted_model = self.model.fit(disp=False)
        return self.fitted_model

    def predict(self, steps=None):
        if self.fitted_model is None:
            raise ValueError("Сначала обучите модель с помощью fit()")
        steps = len(self.test) if steps is None else steps
        self.predictions = self.fitted_model.forecast(steps=steps)
        return self.predictions

    #метрики
    def evaluate(self):
        if self.predictions is None:
            raise ValueError("Сначала сделайте прогноз с помощью predict()")
        rmse = np.sqrt(mean_squared_error(self.test, self.predictions[:len(self.test)]))
        mape = mean_absolute_percentage_error(self.test, self.predictions[:len(self.test)])
        mae = mean_absolute_error(self.test, self.predictions[:len(self.test)])
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}")
        return {'mae': mae, 'rmse': rmse, 'mape': mape}

    def plot_results(self, title="Прогноз ARIMA/SARIMA"):
        if self.predictions is None:
            raise ValueError("Сначала сделайте прогноз с помощью predict()")
        plt.figure(figsize=(12, 6))
        # Отображаем последние 100 точек обучающих данных
        plt.plot(self.train.index[-100:], self.train[-100:], label='Тренировочные данные', color='blue')
        # Тестовые данные
        plt.plot(self.test.index, self.test, label='Тестовые данные', color='orange')
        # Прогноз
        forecast_index = pd.date_range(start=self.test.index[0], periods=len(self.predictions), freq='D')
        plt.plot(forecast_index, self.predictions, label='Прогноз', color='red')
        plt.legend()
        plt.title(title)
        plt.xlabel('Дата')
        plt.ylabel('Значение')
        plt.grid(True)
        plt.show()

# Тестирование на разных наборах данных
def test_arima_on_examples():
    # Список тестовых примеров
    test_cases = [
        {
            "name": "Сильный тренд и слабая сезонность",
            "trend_slope": 0.2,
            "seasonality_period": 7,
            "seasonality_amplitude": 2,
            "noise_scale": 1
        },
        {
            "name": "Слабый тренд и сильная сезонность",
            "trend_slope": 0.05,
            "seasonality_period": 14,
            "seasonality_amplitude": 10,
            "noise_scale": 1
        },
        {
            "name": "Нет тренда, только сезонность и шум",
            "trend_slope": 0,
            "seasonality_period": 7,
            "seasonality_amplitude": 5,
            "noise_scale": 2
        },
        {
            "name": "Очень шумный ряд с месячной сезонностью",
            "trend_slope": 0.01,
            "seasonality_period": 30,
            "seasonality_amplitude": 3,
            "noise_scale": 5
        },
        {
            "name": "Ряд с экспоненциальным трендом и слабой сезонностью",
            "trend_type": "exp",
            "trend_slope": 0.1,
            "seasonality_period": 7,
            "seasonality_amplitude": 2,
            "noise_scale": 1
        },
        {
            "name": "Сильная месячная сезонность без тренда",
            "trend_type": "none",
            "trend_slope": 0,
            "seasonality_period": 30,
            "seasonality_amplitude": 15,
            "noise_scale": 0.5
        },
        {
            "name": "Шумный ряд с длинной сезонностью",
            "trend_type": "linear",
            "trend_slope": 0.02,
            "seasonality_period": 90,
            "seasonality_amplitude": 5,
            "noise_scale": 3
        },
        {
            "name": "Стационарный ряд с минимальной сезонностью",
            "trend_type": "none",
            "trend_slope": 0,
            "seasonality_period": 7,
            "seasonality_amplitude": 0.5,
            "noise_scale": 1
        }
    ]

    # Параметры для поиска
    p_range, d_range, q_range = range(0, 2), range(0, 2), range(0, 2)
    P_range, D_range, Q_range = range(0, 2), range(0, 2), range(0, 2)

    for case in test_cases:
        print(f"\n=== Тест: {case['name']} ===")
        # Генерация данных
        data = generate_test_data(
            periods=200,
            trend_slope=case["trend_slope"],
            seasonality_period=case["seasonality_period"],
            seasonality_amplitude=case["seasonality_amplitude"],
            noise_scale=case["noise_scale"],
            trend_type=case.get("trend_type", "linear")  # Учитывать trend_type
        )

        # Инициализация и тестирование модели
        forecaster = EnhancedARIMAForecaster(data, train_size=0.9)
        forecaster.check_stationarity()
        forecaster.grid_search_params(p_range, d_range, q_range, seasonal=True,
                                      P_range=P_range, D_range=D_range, Q_range=Q_range,
                                      m=case["seasonality_period"])
        forecaster.fit()
        forecaster.predict(steps=1000)
        forecaster.evaluate()
        forecaster.plot_results(title=f"Прогноз для: {case['name']}")

if __name__ == "__main__":
    np.random.seed(42)
    test_arima_on_examples()
