import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter
import torch.distributions as dist


def load_sunspot_data():
    """Загружает, очищает и обрабатывает данные о солнечных пятнах из CSV."""
    try:
        df = pd.read_csv("monthly-sunspots.csv")
        df.columns = ["Month", "Sunspots"]
        df["Month"] = pd.to_datetime(df["Month"], errors='coerce')
        df["Sunspots"] = pd.to_numeric(df["Sunspots"], errors="coerce")
        df = df.dropna()

        # Очистка выбросов
        Q1 = df["Sunspots"].quantile(0.25)
        Q3 = df["Sunspots"].quantile(0.75)
        IQR = Q3 - Q1
        df["Sunspots"] = df["Sunspots"].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

        # Сглаживание с фильтром Ходрика-Прескотта
        series = hpfilter(df["Sunspots"].values, lamb=1600 * 12)[1]  # Используем тренд
        dates = df["Month"].values
        return series, dates
    except FileNotFoundError:
        print("Error: 'monthly-sunspots.csv' not found.")
        exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)


class DeepAR(nn.Module):
    """Модель DeepAR для вероятностного прогнозирования временных рядов."""

    def __init__(self, input_size, hidden_size, num_layers, forecast_horizon, dropout=0.1):
        super(DeepAR, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon

        # LSTM для обработки временного ряда
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Полносвязные слои для предсказания параметров нормального распределения
        self.mu = nn.Linear(hidden_size, forecast_horizon)  # Среднее
        self.sigma = nn.Linear(hidden_size, forecast_horizon)  # Стандартное отклонение

    def forward(self, x, hidden=None):
        # x: (batch_size, seq_len, input_size)
        lstm_out, hidden = self.lstm(x, hidden)  # lstm_out: (batch_size, seq_len, hidden_size)

        # Используем последний выход LSTM для предсказания
        last_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # Предсказываем параметры нормального распределения
        mu = self.mu(last_out)  # (batch_size, forecast_horizon)
        sigma = torch.exp(self.sigma(last_out))  # (batch_size, forecast_horizon), экспонента для положительного sigma

        return mu, sigma, hidden

    def loss(self, mu, sigma, target):
        """Отрицательная логарифмическая потеря правдоподобия при нормальном распределении."""
        distribution = dist.Normal(mu, sigma + 1e-6)  # Добавляем малую константу для стабильности
        log_prob = distribution.log_prob(target)
        return -torch.mean(log_prob)


def create_lagged_features(series, lags=[12, 24, 132]):
    """Создает отложенные функции для серии."""
    X = []
    for i in range(max(lags), len(series)):
        features = [series[i - lag] for lag in lags]
        X.append(features)
    return np.array(X)


def train_deepar(series, input_size, forecast_horizon, lags=[12, 24, 132], epochs=200, hidden_size=64, num_layers=2):
    """Обучает модель DeepAR с запаздывающими функциями."""
    try:
        mean = np.mean(series)
        std = np.std(series) if np.std(series) != 0 else 1.0
        series_norm = (series - mean) / std
        lagged_features = create_lagged_features(series_norm, lags)

        # Подготовка данных
        X, y = [], []
        offset = max(lags)
        for i in range(offset, len(series_norm) - input_size - forecast_horizon):
            X.append(np.concatenate([series_norm[i:i + input_size], lagged_features[i - offset]]))
            y.append(series_norm[i + input_size:i + input_size + forecast_horizon])
        if not X:
            raise ValueError("Insufficient data for training windows")
        X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)  # (batch, seq_len, 1)
        y = torch.tensor(np.array(y), dtype=torch.float32)

        # Инициализация модели
        model = DeepAR(input_size=input_size + len(lags), hidden_size=hidden_size, num_layers=num_layers,
                       forecast_horizon=forecast_horizon)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        # Обучение
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            mu, sigma, _ = model(X)
            loss = model.loss(mu, sigma, y)
            loss.backward()
            optimizer.step()

        # Прогноз
        model.eval()
        with torch.no_grad():
            last_window = np.concatenate([series_norm[-input_size:], lagged_features[-1]])
            last_window = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            mu, _, _ = model(last_window)

        return mu.squeeze().numpy() * std + mean
    except Exception as e:
        print(f"Error in DeepAR: {e}")
        return np.zeros(forecast_horizon)


def compute_metrics(true_values, predicted_values):
    """Вычисляет MAE, RMSE и MAPE."""
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mape = mean_absolute_percentage_error(true_values, predicted_values + 1e-10)
    return mae, rmse, mape


def plot_results(series, true_values, deepar_pred, split_point, forecast_horizon, dates):
    """Визуализация"""
    plt.figure(figsize=(12, 6))
    plt.plot(dates[split_point - 100:split_point], series[split_point - 100:split_point],
             label="Training Data", color="blue")
    plt.plot(dates[split_point:split_point + forecast_horizon], true_values,
             label="Test Data", color="orange")
    plt.plot(dates[split_point:split_point + forecast_horizon], deepar_pred,
             label="DeepAR Forecast", color="purple")
    plt.title("Sunspot Number Forecast")
    plt.xlabel("Date")
    plt.ylabel("Sunspot Number")
    plt.legend()
    plt.grid(True)
    plt.savefig('sunspot_forecast.png')
    plt.close()


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    forecast_horizon = 12
    input_size = 132

    # Загрузка и обработка данных
    series, dates = load_sunspot_data()
    split_point = len(series) - forecast_horizon

    # Разделение данных
    train_series = series[:split_point]
    true_values = series[split_point:split_point + forecast_horizon]

    # DeepAR прогноз
    deepar_pred = train_deepar(train_series, input_size, forecast_horizon)

    # Вычислять показатели
    deepar_mae, deepar_rmse, deepar_mape = compute_metrics(true_values, deepar_pred)

    # Print results
    print(f"DeepAR MAE: {deepar_mae:.4f}, RMSE: {deepar_rmse:.4f}, MAPE: {deepar_mape:.4f}")
    print(f"DeepAR forecast (first 5): {deepar_pred[:5]}")

    # Visualize results
    plot_results(series, true_values, deepar_pred, split_point, forecast_horizon, dates)