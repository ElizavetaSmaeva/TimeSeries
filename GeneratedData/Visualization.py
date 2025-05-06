import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Для PyTorch Forecasting
from pytorch_forecasting import TimeSeriesDataSet

# Параметры генерации
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

# Генератор временного ряда
def generate_series(start_date='2023-01-01', periods=120, freq='D',
                    trend_slope=0.1, seasonality_period=7, seasonality_amplitude=5,
                    noise_scale=1, trend_type='linear', group_id='series_0'):
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    time_idx = np.arange(periods)

    if trend_type == 'exp':
        trend = np.exp(trend_slope * time_idx)
    elif trend_type == 'none':
        trend = np.zeros(periods)
    else:  # linear
        trend = trend_slope * time_idx

    seasonality = seasonality_amplitude * np.sin(2 * np.pi * time_idx / seasonality_period)
    noise = np.random.normal(0, noise_scale, periods)

    value = trend + seasonality + noise

    return pd.DataFrame({
        "time_idx": time_idx,
        "date": dates,
        "series": group_id,
        "value": value
    })

# Собираем все данные
all_series = []
for i, case in enumerate(test_cases):
    df = generate_series(
        trend_slope=case.get("trend_slope", 0.1),
        seasonality_period=case.get("seasonality_period", 7),
        seasonality_amplitude=case.get("seasonality_amplitude", 5),
        noise_scale=case.get("noise_scale", 1),
        trend_type=case.get("trend_type", "linear"),
        group_id=f"series_{i}"
    )
    df["scenario"] = case["name"]
    all_series.append(df)

# Объединение в один DataFrame
combined_df = pd.concat(all_series).reset_index(drop=True)

# Отображение графиков
fig, axs = plt.subplots(len(test_cases), 1, figsize=(12, 3 * len(test_cases)), sharex=True)
for i, case in enumerate(test_cases):
    subset = combined_df[combined_df["scenario"] == case["name"]]
    axs[i].plot(subset["date"], subset["value"])
    axs[i].set_title(case["name"])
    axs[i].grid(True)
plt.tight_layout()
plt.show()

# Показываем первые строки итогового датафрейма
combined_df.head()