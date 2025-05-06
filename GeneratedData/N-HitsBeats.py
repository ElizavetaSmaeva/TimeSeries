import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import pandas as pd

# Проверка зависимостей
try:
    import numpy, matplotlib, torch, statsmodels, scipy, sklearn
except ImportError as e:
    print(
        f"Ошибка: отсутствует библиотека {e.name}. Установите зависимости: pip install numpy matplotlib torch statsmodels scipy scikit-learn")
    sys.exit(1)

# Тестовые данные
test_cases = [
    {
        "name": "Сильный тренд и слабая сезонность",
        "trend_slope": 0.2,
        "seasonality_period": 7,
        "seasonality_amplitude": 2,
        "noise_scale": 1,
        "trend_type": "linear"
    },
    {
        "name": "Слабый тренд и сильная сезонность",
        "trend_slope": 0.05,
        "seasonality_period": 14,
        "seasonality_amplitude": 8,
        "noise_scale": 1,
        "trend_type": "linear"
    },
    {
        "name": "Нет тренда, только сезонность и шум",
        "trend_slope": 0,
        "seasonality_period": 7,
        "seasonality_amplitude": 5,
        "noise_scale": 1.5,
        "trend_type": "none"
    },
    {
        "name": "Шумный ряд с месячной сезонностью",
        "trend_slope": 0.01,
        "seasonality_period": 30,
        "seasonality_amplitude": 3,
        "noise_scale": 2,
        "trend_type": "linear"
    },
    {
        "name": "Ряд с экспоненциальным трендом и слабой сезонностью",
        "trend_type": "exp",
        "trend_slope": 0.05,
        "seasonality_period": 7,
        "seasonality_amplitude": 2,
        "noise_scale": 1
    },
    {
        "name": "Сильная месячная сезонность без тренда",
        "trend_type": "none",
        "trend_slope": 0,
        "seasonality_period": 30,
        "seasonality_amplitude": 10,
        "noise_scale": 0.5
    },
    {
        "name": "Шумный ряд с длинной сезонностью",
        "trend_type": "linear",
        "trend_slope": 0.02,
        "seasonality_period": 30,
        "seasonality_amplitude": 5,
        "noise_scale": 2
    },
    {
        "name": "Стационарный ряд с минимальной сезонностью",
        "trend_type": "none",
        "trend_slope": 0,
        "seasonality_period": 7,
        "seasonality_amplitude": 1,
        "noise_scale": 1
    }
]

def generate_time_series(length, case, start_date='2023-01-01', freq='D'):
    """Генерирует временной ряд на основе параметров тестового случая и возвращает даты."""
    t = np.arange(length)
    series = np.zeros(length)

    if case["trend_type"] == "linear":
        series += case["trend_slope"] * t
    elif case["trend_type"] == "exp":
        series += np.exp(case["trend_slope"] * t / length) - 1

    seasonality = case["seasonality_amplitude"] * np.sin(2 * np.pi * t / case["seasonality_period"])
    series += seasonality

    noise = np.random.normal(0, case["noise_scale"], length)
    series += noise

    # Нормализация: приводим значения к диапазону, похожему на ARi.py
    series = (series - series.min()) / (series.max() - series.min()) * 60 - 10

    dates = pd.date_range(start=start_date, periods=length, freq=freq)
    return series, dates

def n_hits_forecast(series, period, forecast_horizon=10):
    """Реализация N-hits: разложение ряда и прогнозирование."""
    try:
        decomposition = seasonal_decompose(series, period=int(period), model='additive', extrapolate_trend='freq')
        trend = decomposition.trend
        seasonal = decomposition.seasonal

        trend_slope = (trend[-1] - trend[-period]) / period
        trend_forecast = trend[-1] + trend_slope * np.arange(1, forecast_horizon + 1)
        seasonal_forecast = np.tile(seasonal[-period:], forecast_horizon // period + 1)[:forecast_horizon]

        return trend_forecast + seasonal_forecast
    except Exception as e:
        print(f"Ошибка в N-hits для периода {period}: {e}")
        return np.zeros(forecast_horizon)

class NBeatsBlock(nn.Module):
    """Блок N-BEATS с базисными функциями."""
    def __init__(self, input_size, forecast_horizon, hidden_size, theta_size):
        super(NBeatsBlock, self).__init__()
        self.input_size = input_size
        self.forecast_horizon = forecast_horizon
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, theta_size)
        )
        self.backcast_basis = nn.Linear(theta_size, input_size)
        self.forecast_basis = nn.Linear(theta_size, forecast_horizon)

    def forward(self, x):
        theta = self.fc(x)
        backcast = self.backcast_basis(theta)
        forecast = self.forecast_basis(theta)
        return backcast, forecast

class NBeatsModel(nn.Module):
    """Модель N-BEATS с несколькими блоками."""
    def __init__(self, input_size, forecast_horizon, hidden_size=128, stacks=2, blocks_per_stack=3):
        super(NBeatsModel, self).__init__()
        self.forecast_horizon = forecast_horizon
        self.blocks = nn.ModuleList()

        for _ in range(stacks):
            for _ in range(blocks_per_stack):
                self.blocks.append(NBeatsBlock(input_size, forecast_horizon, hidden_size, theta_size=hidden_size))

    def forward(self, x):
        forecast = torch.zeros(x.shape[0], self.forecast_horizon).to(x.device)
        backcast = x

        for block in self.blocks:
            block_backcast, block_forecast = block(backcast)
            backcast = backcast - block_backcast
            forecast = forecast + block_forecast

        return backcast, forecast

def train_n_beats(series, input_size, forecast_horizon, epochs=200):
    """Обучение модели N-BEATS."""
    try:
        mean = np.mean(series)
        std = np.std(series)
        if std == 0:
            std = 1.0
        series = (series - mean) / std
        series = torch.tensor(series, dtype=torch.float32).unsqueeze(0)

        X, y = [], []
        for i in range(len(series[0]) - input_size - forecast_horizon):
            X.append(series[:, i:i + input_size])
            y.append(series[:, i + input_size:i + input_size + forecast_horizon])
        if not X:
            raise ValueError("Недостаточно данных для создания обучающих окон")
        X = torch.cat(X, dim=0)
        y = torch.cat(y, dim=0)

        model = NBeatsModel(input_size=input_size, forecast_horizon=forecast_horizon)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            _, forecast = model(X)
            loss = criterion(forecast, y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            last_window = series[:, -input_size:].clone()
            _, forecast = model(last_window)

        forecast = forecast.squeeze().numpy() * std + mean
        return forecast
    except Exception as e:
        print(f"Ошибка в N-BEATS: {e}")
        return np.zeros(forecast_horizon)

def compute_metrics(true_values, predicted_values):
    """Вычисляет MAE, RMSE и MAPE."""
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mape = mean_absolute_percentage_error(true_values, predicted_values)
    return mae, rmse, mape

def plot_results(case_name, series, true_values, n_hits_pred, n_beats_pred, split_point, forecast_horizon, dates):
    """Визуализация исходного ряда, реальных значений и прогнозов с использованием дат."""
    plt.figure(figsize=(12, 6))
    plt.plot(dates[split_point - 100:split_point], series[split_point - 100:split_point],
             label="Тренировочные данные", color="blue")
    plt.plot(dates[split_point:split_point + forecast_horizon], true_values,
             label="Тестовые данные", color="orange")
    plt.plot(dates[split_point:split_point + forecast_horizon], n_hits_pred,
             label="N-hits прогноз", color="green")
    plt.plot(dates[split_point:split_point + forecast_horizon], n_beats_pred,
             label="N-BEATS прогноз", color="red")
    plt.title(f"Прогноз для: {case_name}")
    plt.xlabel("Дата")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid(True)
    plt.show()

# Основной цикл
if __name__ == "__main__":
    np.random.seed(42)
    length = 1000
    forecast_horizon = 100
    input_size = 100
    split_point = length - forecast_horizon  # 900 точек для обучения, 100 для теста

    for case in test_cases:
        print(f"\nОбработка: {case['name']}")

        # Генерация ряда с датами
        series, dates = generate_time_series(length, case)

        # Разделение на обучение и тест
        train_series = series[:split_point]
        true_values = series[split_point:split_point + forecast_horizon]

        # Прогноз N-hits
        n_hits_pred = n_hits_forecast(train_series, case["seasonality_period"], forecast_horizon)

        # Прогноз N-BEATS
        n_beats_pred = train_n_beats(train_series, input_size, forecast_horizon)

        # Вычисление метрик
        n_hits_mae, n_hits_rmse, n_hits_mape = compute_metrics(true_values, n_hits_pred)
        n_beats_mae, n_beats_rmse, n_beats_mape = compute_metrics(true_values, n_beats_pred)

        print(f"N-hits MAE: {n_hits_mae:.4f}, RMSE: {n_hits_rmse:.4f}, MAPE: {n_hits_mape:.4f}")
        print(f"N-BEATS MAE: {n_beats_mae:.4f}, RMSE: {n_beats_rmse:.4f}, MAPE: {n_beats_mape:.4f}")

        # Вывод первых 5 точек прогноза
        print(f"N-hits прогноз: {n_hits_pred[:5]}")
        print(f"N-BEATS прогноз: {n_beats_pred[:5]}")

        # Визуализация
        plot_results(case["name"], series, true_values, n_hits_pred, n_beats_pred, split_point, forecast_horizon, dates)