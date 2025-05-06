import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_sunspot_data():
    """Загрузка и логарифмирование данных о солнечных пятнах"""
    try:
        df = pd.read_csv("monthly-sunspots.csv")
        df.columns = ["Month", "Sunspots"]
        df["Month"] = pd.to_datetime(df["Month"], errors='coerce')
        df["Sunspots"] = pd.to_numeric(df["Sunspots"], errors="coerce")
        df = df.dropna()

        df["Sunspots"] = np.log1p(df["Sunspots"])
        return pd.Series(df["Sunspots"].values, index=df["Month"])
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None


class EnhancedSARIMAForecaster:
    def __init__(self, data, train_size=0.9):
        self.data = data
        self.train_size = train_size
        self.train, self.test = self._split_data()
        self.model = None
        self.fitted_model = None
        self.predictions = None
        self.best_params = None
        self.best_seasonal_params = None

    def _split_data(self):
        """Разделение данных на обучающую и тестовую выборки"""
        split_point = int(len(self.data) * self.train_size)
        return self.data[:split_point], self.data[split_point:]

    def check_stationarity(self):
        """Проверка стационарности ряда"""
        try:
            result = adfuller(self.data)
            print('ADF Statistic:', result[0])
            print('p-value:', result[1])
            print('Critical Values:', result[4])
            if result[1] <= 0.05:
                print("Ряд стационарен (p-value <= 0.05)")
            else:
                print("Ряд нестационарен (p-value > 0.05). Используем d=1 или SARIMA.")
        except Exception as e:
            print(f"Ошибка при проверке стационарности: {e}")

    def grid_search_params(self, p_range, d_range, q_range, P_range, D_range, Q_range, m):
        """Поиск оптимальных параметров SARIMA"""
        best_score = float('inf')
        best_params = None
        best_seasonal_params = None

        # Ограничение числа комбинаций для экономии памяти
        for p, d, q in itertools.product(p_range, d_range, q_range):
            for P, D, Q in itertools.product(P_range, D_range, Q_range):
                try:
                    model = SARIMAX(self.train, order=(p, d, q), seasonal_order=(P, D, Q, m),
                                    enforce_stationarity=False, enforce_invertibility=False)
                    results = model.fit(maxiter=50)  # Ограничение итераций
                    aic = results.aic
                    print(f"SARIMA ({p},{d},{q}) x ({P},{D},{Q},{m}) AIC: {aic:.2f}")
                    if aic < best_score:
                        best_score = aic
                        best_params = (p, d, q)
                        best_seasonal_params = (P, D, Q, m)
                except Exception as e:
                    print(f"Ошибка для SARIMA ({p},{d},{q}) x ({P},{D},{Q},{m}): {e}")
                    continue

        self.best_params = best_params
        self.best_seasonal_params = best_seasonal_params
        print(f"\nЛучшие параметры (p,d,q): {best_params}")
        print(f"Лучшие сезонные параметры (P,D,Q,m): {best_seasonal_params}")
        print(f"Лучший AIC: {best_score:.2f}")
        return best_params, best_seasonal_params

    def fit(self, order=None, seasonal_order=None):
        """Обучение модели SARIMA"""
        try:
            if order is None and self.best_params is not None:
                order = self.best_params
                seasonal_order = self.best_seasonal_params
            elif order is None:
                raise ValueError("Укажите параметры модели или выполните grid_search_params.")

            self.model = SARIMAX(self.train, order=order, seasonal_order=seasonal_order,
                                 enforce_stationarity=False, enforce_invertibility=False)
            self.fitted_model = self.model.fit(maxiter=50)  # Убрано disp, добавлено maxiter
            return self.fitted_model
        except Exception as e:
            print(f"Ошибка при обучении модели: {e}")
            return None

    def predict(self, steps=None):
        """Прогнозирование"""
        try:
            if self.fitted_model is None:
                raise ValueError("Сначала обучите модель с помощью fit()")
            steps = len(self.test) if steps is None else steps
            self.predictions = self.fitted_model.forecast(steps=steps)
            return self.predictions
        except Exception as e:
            print(f"Ошибка при прогнозировании: {e}")
            return None

    def evaluate(self):
        """Оценка качества модели"""
        try:
            if self.predictions is None:
                raise ValueError("Сначала сделайте прогноз с помощью predict()")
            pred = np.expm1(self.predictions)
            true = np.expm1(self.test[:len(pred)])
            rmse = np.sqrt(mean_squared_error(true, pred))
            mape = mean_absolute_percentage_error(true, pred)
            mae = mean_absolute_error(true, pred)
            print(f"MAE: {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAPE: {mape:.2f}")
            return {'mae': mae, 'rmse': rmse, 'mape': mape}
        except Exception as e:
            print(f"Ошибка при оценке модели: {e}")
            return {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan}

    def plot_results(self, title="Прогноз SARIMA"):
        """Визуализация результатов"""
        try:
            if self.predictions is None:
                raise ValueError("Сначала сделайте прогноз с помощью predict()")
            plt.figure(figsize=(12, 6))
            plt.plot(self.train.index[-100:], np.expm1(self.train[-100:]), label='Тренировочные данные', color='blue')
            plt.plot(self.test.index, np.expm1(self.test), label='Тестовые данные', color='orange')
            forecast_index = pd.date_range(start=self.test.index[0], periods=len(self.predictions), freq='M')
            plt.plot(forecast_index, np.expm1(self.predictions), label='Прогноз', color='red')
            plt.title(title)
            plt.xlabel("Дата")
            plt.ylabel("Солнечные пятна")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Ошибка при визуализации: {e}")


def test_sarima_on_sunspots():
    """Тестирование SARIMA на данных о солнечных пятнах"""
    data = load_sunspot_data()
    if data is None:
        return

    # Параметры для поиска
    p_range = range(0, 3)
    d_range = range(0, 2)  # Учтена стационарность
    q_range = range(0, 3)
    P_range = range(0, 2)
    D_range = range(0, 2)
    Q_range = range(0, 2)
    m = 120  # 10 лет (120 месяцев) для соответствия солнечному циклу

    print("\n=== SARIMA ===")
    forecaster_sarima = EnhancedSARIMAForecaster(data)
    forecaster_sarima.check_stationarity()
    forecaster_sarima.grid_search_params(
        p_range, d_range, q_range,
        P_range=P_range, D_range=D_range, Q_range=Q_range, m=m
    )
    forecaster_sarima.fit()
    forecaster_sarima.predict()
    metrics = forecaster_sarima.evaluate()
    forecaster_sarima.plot_results("SARIMA прогноз")

    # Сохранение результатов
    results = pd.DataFrame({
        'Model': ['SARIMA'],
        'MAE': [metrics['mae']],
        'RMSE': [metrics['rmse']],
        'MAPE': [metrics['mape']]
    })
    results.to_csv("sarima_results.csv", index=False)
    print("\nРезультаты сохранены в 'sarima_results.csv'")


if __name__ == "__main__":
    np.random.seed(42)
    test_sarima_on_sunspots()