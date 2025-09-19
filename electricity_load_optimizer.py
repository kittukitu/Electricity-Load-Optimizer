import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Generate synthetic electricity load dataset ---
np.random.seed(42)
days = 730  # Two years of daily data
dates = pd.date_range(start="2023-01-01", periods=days, freq='D')

# Simulate daily load (MW) with seasonality and trend (higher in winter/summer, weekdays/weekends)
base_load = 1500 + np.linspace(0, 100, days)  # slight upward trend
seasonality_annual = 300 * np.sin(2 * np.pi * dates.dayofyear / 365)
seasonality_weekly = 100 * ((dates.dayofweek < 5).astype(int))  # Weekdays heavier consumption
random_noise = np.random.normal(0, 50, days)

load = base_load + seasonality_annual + seasonality_weekly + random_noise
load = np.clip(load, 500, None)  # Load can’t be negative or too low

# Additional features: temperature (°F), holiday indicator
temperature = 60 + 20 * np.sin(2 * np.pi * (dates.dayofyear + 180) / 365) + np.random.normal(0, 3, days)  # Inverse seasonality
holiday = ((dates.month == 12) & (dates.day >= 24) & (dates.day <= 26)).astype(int)  # Christmas period

data = pd.DataFrame({
    'date': dates,
    'load_MW': load,
    'temperature_F': temperature,
    'holiday': holiday
})

# Save dataset
data.to_csv("electricity_load_data.csv", index=False)
print("Synthetic electricity load dataset saved as 'electricity_load_data.csv'.")

# --- Step 2: Feature engineering ---
data['day_of_week'] = data['date'].dt.dayofweek
data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
data['load_lag1'] = data['load_MW'].shift(1).bfill()
data['load_lag7'] = data['load_MW'].shift(7).bfill()

features = ['temperature_F', 'holiday', 'is_weekend', 'load_lag1', 'load_lag7']
target = 'load_MW'

X = data[features]
y = data[target]

# --- Step 3: Train/test split ---
train_size = int(0.85 * len(data))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# --- Step 4: Train XGBoost regressor ---
model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# --- Step 5: Evaluate model ---
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.2f} MW")

# --- Step 6: User input for next day features and load prediction ---
print("\nEnter next day details for load prediction:")

def input_float(prompt, default):
    try:
        val = input(f"{prompt} (e.g. {default:.1f}): ")
        return float(val) if val.strip() != '' else default
    except ValueError:
        print("Invalid input. Using default.")
        return default

def input_int(prompt, default):
    try:
        val = input(f"{prompt} (0 or 1, default {default}): ")
        return int(val) if val.strip() != '' else default
    except ValueError:
        print("Invalid input. Using default.")
        return default

# Use latest known values as defaults
default_temp = data['temperature_F'].iloc[-1]
default_holiday = 0
default_weekend = 0

next_day_temp = input_float("Temperature (°F)", default_temp)
next_day_holiday = input_int("Holiday? (1 for yes, 0 for no)", default_holiday)
next_day_weekend = input_int("Weekend? (1 for yes, 0 for no)", default_weekend)

latest_features = X.iloc[-1:].copy()
latest_features['load_lag1'] = data['load_MW'].iloc[-1]
latest_features['load_lag7'] = data['load_MW'].iloc[-7]
latest_features['temperature_F'] = next_day_temp
latest_features['holiday'] = next_day_holiday
latest_features['is_weekend'] = next_day_weekend

next_day_load_pred = model.predict(latest_features)[0]
print(f"\nPredicted electricity load for next day: {next_day_load_pred:.2f} MW")

# --- Step 7: Simple supply optimization and cost estimation ---
supply_buffer = 0.1  # 10% buffer
optimal_supply = next_day_load_pred * (1 + supply_buffer)

cost_per_mw = 100  # example cost per MW in USD
estimated_cost = optimal_supply * cost_per_mw

print(f"Recommended supply with 10% buffer: {optimal_supply:.2f} MW")
print(f"Estimated cost for supply: ${estimated_cost:,.2f}")

# --- Step 8: Visualization ---
plt.figure(figsize=(14,7))
plt.plot(data['date'], data['load_MW'], label='Actual Load')
plt.plot(data['date'].iloc[train_size:], y_pred, label='Predicted Load')
plt.xlabel('Date')
plt.ylabel('Electricity Load (MW)')
plt.title('Electricity Load Forecast vs Actual')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(x=data['temperature_F'], y=data['load_MW'], hue=data['is_weekend'], palette='Set1')
plt.xlabel('Temperature (°F)')
plt.ylabel('Electricity Load (MW)')
plt.title('Load vs Temperature (Weekday vs Weekend)')
plt.tight_layout()
plt.show()
