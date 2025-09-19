Electricity Load Forecasting and Supply Optimization
This project creates a synthetic electricity load dataset with daily data for two years and builds an XGBoost regression model to forecast next-day electricity load. It features user input for custom next-day conditions, supply optimization recommendations with cost estimation, and visualization of load patterns and forecast accuracy.

Features
Synthetic daily electricity load data considering trend, annual and weekly seasonality, plus random noise.

Additional features: temperature (°F), holiday indicator, weekend indicator, and lagged load values.

XGBoost regression model to predict next-day load based on engineered features.

User input interface for custom next-day temperature, holiday, and weekend flags.

Supply recommendation with buffer percentage and estimated cost.

Visualizations:

Actual vs predicted load over time

Load relationship with temperature by weekday/weekend.

Requirements
Python 3.x

pandas

numpy

xgboost

scikit-learn

matplotlib

seaborn

Install dependencies:

bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn
Usage
Run the script
The script generates the synthetic electricity_load_data.csv dataset and trains the forecasting model.

Input next-day features
When prompted, enter values or accept defaults for:

Temperature (°F)

Holiday flag (1 for yes, 0 for no)

Weekend flag (1 for yes, 0 for no)

Output

Predicted electricity load (MW) for the next day

Recommended supply quantity with a 10% safety buffer

Estimated cost for this supply (based on a fixed cost per MW)

Visualizations

Line plot comparing actual load and predicted load over the test period

Scatter plot showing load vs temperature split by weekday/weekend

Data Description
date: Daily date index for two years

load_MW: Simulated electricity load in megawatts

temperature_F: Simulated daily temperature in Fahrenheit

holiday: Binary flag for Christmas period (Dec 24-26)

day_of_week: Numeric day of week (0=Monday)

is_weekend: Binary weekend flag

load_lag1, load_lag7: Lagged electricity load from previous day and week

Notes
The train/test split respects temporal order to avoid leakage.

The cost per MW and buffer percent can be adjusted in the script.

Visualizations require a graphical environment for display.