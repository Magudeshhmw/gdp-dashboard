from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from datetime import datetime
import numpy as np
import io
import base64

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        company = request.form["company"]
        start_date = request.form["start_date"]
        end_date = request.form["end_date"]

        # Convert input strings to datetime objects
        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            return render_template("index.html", error_message="Error: Date format is incorrect. Please use YYYY-MM-DD.")

        # Download historical data for the given company
        data = yf.download(company, start=start_date, end=end_date)

        if data.empty:
            return render_template("index.html", error_message="No data available for this symbol or date range.")

        # ARIMA Model Forecasting
        df = data[['Close']]
        train_size = int(len(df) * 0.8)
        train, test = df.iloc[:train_size], df.iloc[train_size:]
        model = ARIMA(train['Close'], order=(1, 1, 2))  # (p, d, q)
        model_fit = model.fit()

        # Forecast
        forecast = model_fit.forecast(steps=len(test))

        # Calculate RMSE manually
        rmse = np.sqrt(mean_squared_error(test['Close'], forecast))

        # Calculate percentage change
        data['Close_pct_change'] = data['Close'].pct_change() * 100

        # Plotting forecast
        plt.figure(figsize=(14, 7))
        plt.plot(train.index, train['Close'], label='Train', color='#203147')
        plt.plot(test.index, test['Close'], label='Test', color='#01ef63')
        plt.plot(test.index, forecast, label='Forecast', color='orange')
        plt.title(f'{company} Close Price Forecast')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()

        # Save forecast plot to a string
        img1 = io.BytesIO()
        plt.savefig(img1, format='png')
        img1.seek(0)
        forecast_plot_url = base64.b64encode(img1.getvalue()).decode('utf8')

        # Plotting percentage change
        plt.figure(figsize=(14, 7))
        plt.plot(data.index, data['Close_pct_change'], label=f'{company} Close % Change', color='blue')
        plt.axhline(0, color='red', linestyle='--', linewidth=1)

        # Correlation Analysis between the user-specified company and AAPL
        company2 = 'AAPL'  # Hardcoded Apple (AAPL)
        
        # Download historical data for AAPL
        data2 = yf.download(company2, start=start_date, end=end_date)
        data2['Close_pct_change'] = data2['Close'].pct_change() * 100
        
        # Plot AAPL's percentage change in close price
        plt.plot(data2.index, data2['Close_pct_change'], label=f'{company2} Close % Change', color='orange')

        plt.title(f'{company} vs {company2} Close Price Percentage Change')
        plt.xlabel('Date')
        plt.ylabel('Percentage Change (%)')
        plt.legend()

        # Save percentage change plot to a string
        img2 = io.BytesIO()
        plt.savefig(img2, format='png')
        img2.seek(0)
        pct_change_plot_url = base64.b64encode(img2.getvalue()).decode('utf8')

        # Return the plots and RMSE
        return render_template(
            "index.html",
            forecast_plot_url=forecast_plot_url,
            pct_change_plot_url=pct_change_plot_url,
            rmse=rmse,
            company=company
        )

    return render_template("index.html", forecast_plot_url=None, pct_change_plot_url=None)

if __name__ == "__main__":
    app.run(debug=True)
