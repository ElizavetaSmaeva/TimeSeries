import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
import pandas as pd
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf

def load_sunspot_data():
    """Загружает, очищает и обрабатывает данные о солнечных пятнах из CSV."""
    try:

        #df = pd.read_csv("Sunspots.csv")
        df = pd.read_csv("monthly-sunspots.csv")
        df.columns = ["Month", "Sunspots"]
        df["Month"] = pd.to_datetime(df["Month"], errors='coerce')
        df["Sunspots"] = pd.to_numeric(df["Sunspots"], errors="coerce")
        df = df.dropna()

        # Очистка выбросов
        Q1 = df["Sunspots"].quantile(0.25)
        Q3 = df["Sunspots"].quantile(0.75)
        IQR = Q3 - Q1
        df["Sunspots"] = df["Sunspots"].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)

        # Сглаживание с фильтром Ходрика-Прескотта
        series = hpfilter(df["Sunspots"].values, lamb=1600*12)[1]  # Используем тренд
        dates = df["Month"].values
        return series, dates
    except FileNotFoundError:
        print("Error: 'monthly-sunspots.csv' not found.")
        exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

def find_optimal_period(series, max_lag=200):
    """Находит оптимальный период с помощью автокорреляции."""
    acf_values = acf(series, nlags=max_lag, fft=True)
    peaks, _ = find_peaks(acf_values, distance=50)
    if len(peaks) == 0:
        return 132  # Значение по умолчанию
    return peaks[np.argmax(acf_values[peaks])]

def n_hits_forecast(series, forecast_horizon=12):
    """Реализует расширенные N-хиты с адаптивным периодом и нелинейным трендом."""
    try:
        # Разделение на тренировочную и валидационную выборки
        val_size = int(0.1 * len(series))
        train_series = series[:-val_size]
        val_series = series[-val_size:]

        # Нормализация
        mean = np.mean(train_series)
        std = np.std(train_series) if np.std(train_series) != 0 else 1.0
        series_norm = (series - mean) / std
        train_norm = series_norm[:-val_size]
        val_norm = series_norm[-val_size:]

        # Определение оптимального периода
        optimal_period = find_optimal_period(train_norm)
        periods = [optimal_period - 4, optimal_period, optimal_period + 4]

        # Определение фазы
        smoothed_series = hpfilter(series_norm, lamb=1600*12)[1]
        peaks, _ = find_peaks(smoothed_series, distance=100)
        troughs, _ = find_peaks(-smoothed_series, distance=100)
        last_peak = peaks[-1] if len(peaks) > 0 else 0
        last_trough = troughs[-1] if len(troughs) > 0 else 0
        is_decline = last_peak > last_trough

        # Прогнозы для разных периодов и выбор лучшего
        forecasts = []
        val_forecasts = []
        for period in periods:
            decomposition = seasonal_decompose(train_norm, period=int(period), model='additive', extrapolate_trend='freq')
            trend = decomposition.trend
            seasonal = decomposition.seasonal

            # Полиномиальный тренд 2-й степени
            trend_clean = trend[~np.isnan(trend)]
            t = np.arange(len(trend_clean)).reshape(-1, 1)
            t_poly = np.hstack([t, t**2])
            model_trend = LinearRegression()
            model_trend.fit(t_poly, trend_clean)
            t_future = np.arange(len(trend_clean), len(trend_clean) + forecast_horizon).reshape(-1, 1)
            t_future_poly = np.hstack([t_future, t_future**2])
            trend_forecast = model_trend.predict(t_future_poly)

            # Сезонность
            seasonal_forecast = np.tile(seasonal[-period:], forecast_horizon // period + 1)[:forecast_horizon]
            forecast = trend_forecast + seasonal_forecast

            # Прогноз на валидационной выборке (для выбора весов)
            val_trend = model_trend.predict(np.hstack([np.arange(len(trend_clean) - val_size, len(trend_clean)).reshape(-1, 1),
                                                       np.arange(len(trend_clean) - val_size, len(trend_clean)).reshape(-1, 1)**2]))
            val_seasonal = np.tile(seasonal[-period:], val_size // period + 1)[:val_size]
            val_forecast = val_trend + val_seasonal
            val_forecasts.append(val_forecast)
            forecasts.append(forecast)

        # Вычисление весов на основе ошибок на валидационной выборке
        errors = [mean_absolute_error(val_norm, vf) for vf in val_forecasts]
        weights = [1 / (e + 1e-10) for e in errors]  # Обратные ошибки как веса
        weights = np.array(weights) / np.sum(weights)

        # Взвешенное усреднение прогнозов
        forecast = np.average(forecasts, axis=0, weights=weights)

        # Моделирование остатков
        residuals = train_norm - (decomposition.trend + decomposition.seasonal)[len(train_norm) - len(train_norm):]
        residuals = residuals[~np.isnan(residuals)]
        t_res = np.arange(len(residuals)).reshape(-1, 1)
        model_res = LinearRegression()
        model_res.fit(t_res, residuals)
        t_future_res = np.arange(len(residuals), len(residuals) + forecast_horizon).reshape(-1, 1)
        residual_forecast = model_res.predict(t_future_res)

        return (forecast + residual_forecast) * std + mean
    except Exception as e:
        print(f"Error in N-hits: {e}")
        return np.zeros(forecast_horizon)

class NBeatsBlock(nn.Module):
    """N-BEATS с базовыми функциями."""
    def __init__(self, input_size, forecast_horizon, hidden_size, theta_size):
        super(NBeatsBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
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
    """N-BEATS модель с несколькими блоками."""
    def __init__(self, input_size, forecast_horizon, hidden_size=256, stacks=2, blocks_per_stack=3):
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

def create_lagged_features(series, lags=[12, 24, 132]):
    """Создает отложенные функции для серии."""
    X = []
    for i in range(max(lags), len(series)):
        features = [series[i - lag] for lag in lags]
        X.append(features)
    return np.array(X)

def train_n_beats(series, input_size, forecast_horizon, lags=[12, 24, 132], epochs=200):
    """Обучает модель N-BEATS с запаздывающими функциями."""
    try:
        mean = np.mean(series)
        std = np.std(series) if np.std(series) != 0 else 1.0
        series_norm = (series - mean) / std
        lagged_features = create_lagged_features(series_norm, lags)

        X, y = [], []
        offset = max(lags)
        for i in range(offset, len(series_norm) - input_size - forecast_horizon):
            X.append(np.concatenate([series_norm[i:i + input_size], lagged_features[i - offset]]))
            y.append(series_norm[i + input_size:i + input_size + forecast_horizon])
        if not X:
            raise ValueError("Insufficient data for training windows")
        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32)

        model = NBeatsModel(input_size=input_size + len(lags), forecast_horizon=forecast_horizon)
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
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
            last_window = np.concatenate([series_norm[-input_size:], lagged_features[-1]])
            last_window = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0)
            _, forecast = model(last_window)

        return forecast.squeeze().numpy() * std + mean
    except Exception as e:
        print(f"Error in N-BEATS: {e}")
        return np.zeros(forecast_horizon)

def compute_metrics(true_values, predicted_values):
    """Вычисляет MAE, RMSE и MAPE."""
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mape = mean_absolute_percentage_error(true_values, predicted_values + 1e-10)
    return mae, rmse, mape

def plot_results(series, true_values, n_hits_pred, n_beats_pred, split_point, forecast_horizon, dates):
    """Визуализирует ряды, истинные значения и прогнозы."""
    plt.figure(figsize=(12, 6))
    plt.plot(dates[split_point-100:split_point], series[split_point-100:split_point],
             label="Training Data", color="blue")
    plt.plot(dates[split_point:split_point+forecast_horizon], true_values,
             label="Test Data", color="orange")
    plt.plot(dates[split_point:split_point+forecast_horizon], n_hits_pred,
             label="N-hits Forecast", color="green")
    plt.plot(dates[split_point:split_point+forecast_horizon], n_beats_pred,
             label="N-BEATS Forecast", color="red")
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

    # N-hits Прогноз
    n_hits_pred = n_hits_forecast(train_series, forecast_horizon=forecast_horizon)

    # N-BEATS Прогноз
    n_beats_pred = train_n_beats(train_series, input_size, forecast_horizon)

    # Compute metrics
    n_hits_mae, n_hits_rmse, n_hits_mape = compute_metrics(true_values, n_hits_pred)
    n_beats_mae, n_beats_rmse, n_beats_mape = compute_metrics(true_values, n_beats_pred)

    # Print results
    print(f"N-hits MAE: {n_hits_mae:.4f}, RMSE: {n_hits_rmse:.4f}, MAPE: {n_hits_mape:.4f}")
    print(f"N-BEATS MAE: {n_beats_mae:.4f}, RMSE: {n_beats_rmse:.4f}, MAPE: {n_beats_mape:.4f}")
    print(f"N-hits forecast (first 5): {n_hits_pred[:5]}")
    print(f"N-BEATS forecast (first 5): {n_beats_pred[:5]}")

    # Visualize results
    plot_results(series, true_values, n_hits_pred, n_beats_pred, split_point, forecast_horizon, dates)