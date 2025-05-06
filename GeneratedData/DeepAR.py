import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from typing import Optional
import pandas as pd
from sklearn.linear_model import LinearRegression


# Расчет метрик
def calculate_metrics(true_values, predicted_values):
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)

    # MAE
    mae = np.mean(np.abs(true_values - predicted_values))

    # RMSE
    rmse = np.sqrt(np.mean((true_values - predicted_values) ** 2))

    # MAPE (избегаем деления на ноль)
    mask = np.abs(true_values) > 1e-3  # Увеличенный порог
    mape = np.mean(np.abs((true_values[mask] - predicted_values[mask]) / true_values[mask])) * 100 if np.any(mask) else float('inf')

    return mae, rmse, mape


#  Генерация временных признаков
def generate_covariates(length, seasonality_periods=[7, 14, 30, 90]):
    covariates = []
    x = np.arange(length)
    # Полиномиальные признаки для тренда
    covariates.append(x / length)  # Линейный член
    covariates.append((x / length) ** 2)  # Квадратичный член
    # Синусоиды и косинусоиды для сезонности
    for period in seasonality_periods:
        covariates.append(np.sin(2 * np.pi * x / period))
        covariates.append(np.cos(2 * np.pi * x / period))
    return np.stack(covariates, axis=-1).astype(np.float32)


#Извлечение тренда
def detrend_series(series, trend_type='linear'):
    x = np.arange(len(series)).reshape(-1, 1)
    if trend_type == 'exp':
        # Сдвигаем ряд, чтобы все значения были положительными
        min_val = np.min(series)
        shift = abs(min_val) + 1e-6 if min_val <= 0 else 0
        y = np.log(series + shift)
        model = LinearRegression()
        model.fit(x, y)
        trend = np.exp(model.predict(x)) - shift
        detrended = series - trend
        return detrended, lambda idx: np.exp(model.predict(idx.reshape(-1, 1))) - shift
    else:
        # Для линейного тренда
        model = LinearRegression()
        model.fit(x, series)
        trend = model.predict(x)
        detrended = series - trend
        return detrended, lambda idx: model.predict(idx.reshape(-1, 1))


#  DeepAR модель
class DeepARModel(pl.LightningModule):
    def __init__(self, input_size=1, covariate_size=1, hidden_size=256, num_layers=3, dropout=0.2, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_size + covariate_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc_mu = nn.Linear(hidden_size, 1)
        self.fc_sigma = nn.Linear(hidden_size, 1)

    def forward(self, x, covariates):
        x = torch.cat([x, covariates], dim=-1)
        out, _ = self.lstm(x)
        mu = self.fc_mu(out)
        sigma = torch.exp(self.fc_sigma(out))
        return mu, sigma

    def training_step(self, batch, batch_idx):
        x, covariates, y = batch
        mu, sigma = self(x, covariates)
        loss = torch.mean(0.5 * torch.log(2 * torch.pi * sigma ** 2) + (y - mu) ** 2 / (2 * sigma ** 2))
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}



class TimeSeriesDataset(Dataset):
    def __init__(self, series: np.ndarray, covariates: np.ndarray, context_length: int):
        self.series = series.astype(np.float32)
        self.covariates = covariates.astype(np.float32)
        self.context_length = context_length
        if len(self.series) <= self.context_length:
            raise ValueError(f"Series length ({len(self.series)}) must be greater than context_length ({self.context_length})")

    def __len__(self):
        length = len(self.series) - self.context_length
        if length <= 0:
            raise ValueError(f"Dataset is empty: series length ({len(self.series)}) <= context_length ({self.context_length})")
        return length

    def __getitem__(self, idx):
        x = self.series[idx:idx + self.context_length]
        cov = self.covariates[idx:idx + self.context_length]
        y = self.series[idx + 1:idx + self.context_length + 1]
        return torch.tensor(x).unsqueeze(-1), torch.tensor(cov), torch.tensor(y).unsqueeze(-1)


# Генерация временного ряда
def generate_series(length, trend_slope=0, seasonality_period=7, seasonality_amplitude=0, noise_scale=1,
                    trend_type='linear'):
    x = np.arange(length)
    if trend_type == "exp":
        trend = np.exp(trend_slope * x / length)
    elif trend_type == "none":
        trend = np.zeros_like(x)
    else:
        trend = trend_slope * x
    seasonality = seasonality_amplitude * np.sin(2 * np.pi * x / seasonality_period)
    noise = np.random.normal(0, noise_scale, size=length)
    return trend + seasonality + noise


#  Обучение и прогнозирование
def train_and_forecast(series, covariates, trend_type='linear', context_length=90, prediction_length=30, epochs=200,
                       num_samples=50, plot=True, case_name=""):
    # Проверка context_length
    train_length = len(series) - prediction_length
    if context_length >= train_length:
        context_length = max(1, train_length - 1)
        print(f"Warning: context_length adjusted to {context_length} to fit series length")

    # Нормализация series и covariates
    series_mean = np.mean(series[:-prediction_length])
    series_std = np.std(series[:-prediction_length]) + 1e-6
    norm_series = (series - series_mean) / series_std
    norm_covariates = (covariates - np.mean(covariates, axis=0)) / (np.std(covariates, axis=0) + 1e-6)

    # Извлекаем тренд
    detrended_series, trend_func = detrend_series(norm_series, trend_type=trend_type)
    train_series = detrended_series[:-prediction_length]
    train_covariates = norm_covariates[:-prediction_length]
    true_future = series[-prediction_length:]  # Реальные значения (без нормализации)
    future_covariates = norm_covariates[-prediction_length - context_length:]

    # Обучение
    dataset = TimeSeriesDataset(train_series, train_covariates, context_length)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = DeepARModel(input_size=1, covariate_size=covariates.shape[-1], hidden_size=256, num_layers=3, dropout=0.2,
                        lr=1e-4)
    trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False, logger=False, enable_model_summary=False)
    trainer.fit(model, loader)

    # Прогноз
    model.eval()
    all_preds = []
    for _ in range(num_samples):
        x_input = torch.tensor(train_series[-context_length:]).float().unsqueeze(0).unsqueeze(-1)
        cov_input = torch.tensor(future_covariates).float().unsqueeze(0)
        preds = []
        for t in range(prediction_length):
            mu, sigma = model(x_input, cov_input[:, t:t + context_length])
            next_val = mu[0, -1, 0].item()
            preds.append(next_val)
            x_input = torch.cat([x_input[:, 1:], torch.tensor([[[next_val]]])], dim=1)
        all_preds.append(preds)

    all_preds = np.array(all_preds)
    mean_residuals = np.mean(all_preds, axis=0)

    # Обратная нормализация и добавление тренда
    future_indices = np.arange(len(series) - prediction_length, len(series))
    trend_values = trend_func(future_indices) * series_std + series_mean
    mean_preds = mean_residuals * series_std + trend_values

    # Расчет метрик
    mae, rmse, mape = calculate_metrics(true_future, mean_preds)
    print(f"Метрики для {case_name}:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}%")

    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(np.arange(len(series) - prediction_length), series[:-prediction_length], label="Обучающие данные",
                 color='blue')
        plt.plot(np.arange(len(series) - prediction_length, len(series)), true_future, label="Тестовые данные",
                 color='green')
        plt.plot(np.arange(len(series) - prediction_length, len(series)), mean_preds, '--', label="Прогноз (среднее)",
                 color='red')
        std_residuals = np.std(all_preds, axis=0) * series_std
        plt.fill_between(np.arange(len(series) - prediction_length, len(series)),
                         mean_preds - 1.96 * std_residuals,
                         mean_preds + 1.96 * std_residuals,
                         color='red', alpha=0.1, label="Доверительный интервал (95%)")
        plt.title(f"{case_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return mean_preds


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

# Запуск тестов
for case in test_cases:
    print(f"\nТест: {case['name']}")
    length = 200
    case_args = {k: v for k, v in case.items() if k != "name"}
    series = generate_series(length=length, **case_args)
    covariates = generate_covariates(length, seasonality_periods=[7, 14, 30, 90])

    trend_type = case.get("trend_type", "linear")
    forecast = train_and_forecast(series, covariates, trend_type=trend_type, context_length=90, prediction_length=30,
                                  epochs=200, num_samples=50, case_name=case['name'])