import argparse
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import google.generativeai as genai

# -------------------
# Configure Gemini API
# -------------------
genai.configure(api_key="AIzaSyC2EVCSgC-DRWVunkKi7Ro0J1upoN3UglE")  # Replace with your Gemini API key
model = genai.GenerativeModel("gemini-1.5-flash")

# -------------------
# Forecast Function
# -------------------
def forecast_load(consumption_history):
    try:
        if len(consumption_history) < 5:
            print("❌ Error: Please enter at least 5 consumption values.")
            return

        data = pd.Series(consumption_history)

        # Holt-Winters Forecast
        model_hw = ExponentialSmoothing(data, trend="add", seasonal=None)
        model_fit = model_hw.fit()
        forecast_values = model_fit.forecast(3)

        # Analyze trend
        trend = "increasing 📈" if forecast_values.mean() > np.mean(consumption_history) else "decreasing 📉"

        # AI Supply Optimization + Cost Analysis
        prompt = f"""
        You are an energy optimization expert. Based on this electricity consumption history: {consumption_history},
        and forecast demand for the next 3 periods: {list(forecast_values)},
        provide:
        1. Supply optimization strategy (renewable vs non-renewable mix, peak/off-peak planning).
        2. Cost-saving analysis for energy companies and consumers.
        3. Recommendations for reducing peak load demand.
        Keep it concise and practical.
        """
        response = model.generate_content(prompt)
        ai_text = response.text if response else "❌ No AI response"

        # Display results
        print("\n⚡ Electricity Load Forecast & AI Analysis")
        print("-" * 50)
        print(f"Next 3 Forecasted Loads : {list(np.round(forecast_values, 2))}")
        print(f"Trend                   : {trend}\n")
        print("🤖 AI Energy Optimization & Cost Analysis:")
        print(ai_text)

    except Exception as e:
        print(f"❌ Error: {e}")

# -------------------
# CLI Setup
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="⚡ Electricity Load Optimizer (Terminal Version)")
    parser.add_argument("--consumption", type=str, help="Comma-separated past electricity consumption values (at least 5)")

    args = parser.parse_args()

    if args.consumption:
        raw_data = args.consumption
    else:
        raw_data = input("Enter past electricity consumption (comma-separated, at least 5 values): ")

    consumption_history = [float(x.strip()) for x in raw_data.split(",") if x.strip()]
    forecast_load(consumption_history)
