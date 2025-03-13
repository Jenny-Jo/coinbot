import os
os.environ['TIMESFM_BACKEND'] = 'jax'  # Force JAX backend

import timesfm
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tqdm import tqdm

# Move this function definition to the top, after imports
def plot_correlation_matrix(data, ax):
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlation Matrix', fontsize=14)

# Define the start date
start_date = '2020-01-01'
# Get multiple data sources
def get_market_data():
    print("Downloading BTC-USD data...")
    btc = yf.download('BTC-USD', start=start_date)
    
    print("Downloading market indicators...")
    indicators = yf.download(['^TNX', '^IXIC', 'GC=F', '^VIX', 'DX-Y.NYB'], 
                           start=start_date)['Close']
    
    # Create individual series
    series = {
        'BTC': btc['Close'],
        'Treasury': indicators['^TNX'],
        'NASDAQ': indicators['^IXIC'],
        'Gold': indicators['GC=F'],
        'VIX': indicators['^VIX'],
        'DXY': indicators['DX-Y.NYB']
    }
    
    # Get the union of all dates
    all_dates = sorted(set().union(*[s.index for s in series.values()]))
    
    # Create DataFrame with all dates and reindex each series
    df = pd.DataFrame(index=all_dates)
    for name, s in series.items():
        df[name] = s.reindex(all_dates)
    
    # Forward fill missing values (weekends/holidays)
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Print data quality info
    print("\nData Summary:")
    print("Date Range:", df.index[0].strftime('%Y-%m-%d'), "to", df.index[-1].strftime('%Y-%m-%d'))
    print("\nMissing Values before fill:")
    print(df.isnull().sum())
    print("\nShape:", df.shape)
    
    # Print basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    return df

# Get and prepare data
print("Fetching market data...")
market_data = get_market_data()

# Normalize all features
def normalize_features(df):
    normalized_df = pd.DataFrame()
    means = {}
    stds = {}
    
    for column in df.columns:
        mean = df[column].mean()
        std = df[column].std()
        normalized_df[column] = (df[column] - mean) / std
        means[column] = mean
        stds[column] = std
    
    return normalized_df, means, stds

# Normalize the data
normalized_data, means, stds = normalize_features(market_data)

# Prepare input for the model
forecast_input = [
    normalized_data['BTC'].values,
    normalized_data['Treasury'].values,
    normalized_data['NASDAQ'].values,
    normalized_data['Gold'].values,
    normalized_data['VIX'].values,
    normalized_data['DXY'].values
]

# Initialize model with correct architecture for covariates
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="cpu",  # Use CPU for Mac ARM
        per_core_batch_size=32,
        horizon_len=128,
        num_layers=50,
        use_positional_embedding=False,
        context_len=2048,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
    ),
)

# Prepare data for forecast
context_window = 32  # Power of 2 (2^5 = 32)

# Get the last context_window days of data for BTC only (univariate forecast)
btc_input = normalized_data['BTC'].values[-context_window:]

# Generate forecast
point_forecast, quantile_forecast = tfm.forecast(
    inputs=[btc_input],  # Pass BTC series as a list with single element
    freq=[0],  # Daily frequency
    normalize=False  # We already normalized the data
)

# Denormalize BTC predictions
point_forecast_denorm = point_forecast[0] * stds['BTC'] + means['BTC']
quantile_forecast_denorm = quantile_forecast[0] * stds['BTC'] + means['BTC']

# Get dates for predictions
last_date = market_data.index[-1]
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=128, freq='D')

# Print predictions
print("\nPredicted prices (next 5 days):")
for date, price in zip(future_dates[:5], point_forecast_denorm[:5]):
    print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")

# Create a figure with multiple subplots
try:
    plt.style.use('seaborn-v0_8')  # Updated style name for newer versions
except:
    plt.style.use('default')  # Fallback to default style if seaborn style is not available

fig = plt.figure(figsize=(20, 25))  # Made taller for additional plot

# 1. Main Forecast Plot
ax1 = plt.subplot(4, 2, 1)
ax1.plot(market_data.index[-200:], market_data['BTC'][-200:], label='Historical', color='blue')
ax1.plot(future_dates, point_forecast_denorm, label='Forecast', color='red', linestyle='--')
ax1.fill_between(future_dates,
                quantile_forecast_denorm[:, 1],
                quantile_forecast_denorm[:, -1],
                color='red', alpha=0.2, label='90% Confidence')
ax1.set_title('BTC Price Forecast', fontsize=14)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Price (USD)', fontsize=12)
ax1.legend()
ax1.tick_params(axis='x', rotation=45)

# 2. Price Distribution Plot
ax2 = plt.subplot(4, 2, 2)
sns.histplot(market_data['BTC'], bins=50, ax=ax2, color='blue', alpha=0.6, label='Historical')
sns.histplot(point_forecast_denorm, bins=20, ax=ax2, color='red', alpha=0.6, label='Forecast')
ax2.set_title('Price Distribution Comparison', fontsize=14)
ax2.set_xlabel('Price (USD)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.legend()

# 3. Returns Plot
historical_returns = np.diff(market_data['BTC']) / market_data['BTC'][:-1] * 100
forecast_returns = np.diff(point_forecast_denorm) / point_forecast_denorm[:-1] * 100

ax3 = plt.subplot(4, 2, 3)
ax3.plot(market_data.index[-100:], historical_returns[-100:], label='Historical Returns', color='blue')
ax3.plot(future_dates[1:], forecast_returns, label='Forecast Returns', color='red', linestyle='--')
ax3.set_title('Daily Returns (%)', fontsize=14)
ax3.set_xlabel('Date', fontsize=12)
ax3.set_ylabel('Returns (%)', fontsize=12)
ax3.legend()
ax3.tick_params(axis='x', rotation=45)

# 4. Volatility Plot (20-day rolling standard deviation)
historical_vol = pd.Series(historical_returns).rolling(20).std()
forecast_vol = pd.Series(forecast_returns).rolling(20).std()

ax4 = plt.subplot(4, 2, 4)
ax4.plot(market_data.index[-100:], historical_vol[-100:], label='Historical Volatility', color='blue')
ax4.plot(future_dates[20:], forecast_vol[19:], label='Forecast Volatility', color='red', linestyle='--')
ax4.set_title('20-Day Rolling Volatility', fontsize=14)
ax4.set_xlabel('Date', fontsize=12)
ax4.set_ylabel('Volatility', fontsize=12)
ax4.legend()
ax4.tick_params(axis='x', rotation=45)

# 5. Backtesting Plot
ax5 = plt.subplot(4, 2, 5)  # Added new row for 2 more plots

# Perform rolling backtests
window_size = 30  # 30-day prediction windows
backtest_predictions = []
backtest_dates = []
actual_prices = []

for i in tqdm(range(len(market_data['BTC']) - window_size - 30, len(market_data['BTC']) - window_size)):
    # Prepare historical data for this window
    historical_prices = market_data['BTC'][:i]
    historical_normalized = (historical_prices - np.mean(historical_prices)) / np.std(historical_prices)
    
    # Generate forecast
    point_forecast_bt, _ = tfm.forecast(
        [historical_normalized],
        freq=[0]
    )
    
    # Denormalize prediction
    prediction = point_forecast_bt[0][0] * np.std(historical_prices) + np.mean(historical_prices)
    backtest_predictions.append(prediction)
    backtest_dates.append(market_data.index[i])
    actual_prices.append(market_data['BTC'].iloc[i])

# Plot backtesting results
ax5.plot(backtest_dates, actual_prices, label='Actual', color='blue')
ax5.plot(backtest_dates, backtest_predictions, label='Predicted', color='red', linestyle='--')
ax5.set_title('Backtesting Results (1-Day Ahead Predictions)', fontsize=14)
ax5.set_xlabel('Date', fontsize=12)
ax5.set_ylabel('Price (USD)', fontsize=12)
ax5.legend()
ax5.tick_params(axis='x', rotation=45)

# 6. Error Analysis Plot
ax6 = plt.subplot(4, 2, 6)

# Calculate prediction errors
errors = np.array(backtest_predictions) - np.array(actual_prices)
percentage_errors = (errors / np.array(actual_prices)) * 100

# Plot error distribution
sns.histplot(percentage_errors, bins=30, ax=ax6, color='purple', alpha=0.6)
ax6.axvline(x=0, color='red', linestyle='--')
ax6.set_title('Prediction Error Distribution', fontsize=14)
ax6.set_xlabel('Percentage Error (%)', fontsize=12)
ax6.set_ylabel('Frequency', fontsize=12)

# Calculate evaluation metrics
mape = mean_absolute_percentage_error(actual_prices, backtest_predictions) * 100
rmse = np.sqrt(mean_squared_error(actual_prices, backtest_predictions))
mae = mean_absolute_error(actual_prices, backtest_predictions)

# Add evaluation metrics to the analysis text
metrics_text = f"""
Model Analysis Metrics:
- Last Known Price: ${market_data['BTC'].iloc[-1]:,.2f}
- Predicted Price (Next Day): ${point_forecast_denorm[0]:,.2f}
- Predicted Change: {((point_forecast_denorm[0] - market_data['BTC'].iloc[-1]) / market_data['BTC'].iloc[-1] * 100):,.2f}%
- 90% Confidence Range (Next Day): ${quantile_forecast_denorm[0, 1]:,.2f} to ${quantile_forecast_denorm[0, -1]:,.2f}
- Historical Volatility (20D): {historical_vol.iloc[-1]:,.2f}%
- Predicted Volatility (20D): {forecast_vol.iloc[-1]:,.2f}%
"""

metrics_text += f"""
Backtesting Metrics (30-day window):
- Mean Absolute Percentage Error: {mape:.2f}%
- Root Mean Square Error: ${rmse:.2f}
- Mean Absolute Error: ${mae:.2f}
- Error Standard Deviation: {np.std(percentage_errors):.2f}%
"""

# 7. Correlation Matrix
ax7 = plt.subplot(4, 2, 7)  # Add new subplot
plot_correlation_matrix(market_data, ax7)

# 8. Feature Importance Plot (using correlation with BTC)
ax8 = plt.subplot(4, 2, 8)
correlations = market_data.corrwith(market_data['BTC']).sort_values(ascending=True)
correlations.plot(kind='barh', ax=ax8)
ax8.set_title('Feature Correlation with BTC', fontsize=14)
ax8.set_xlabel('Correlation Coefficient', fontsize=12)

# Add feature correlation info to metrics text
metrics_text += "\nFeature Correlations with BTC:"
for feature, corr in correlations.items():
    if feature != 'BTC':
        metrics_text += f"\n- {feature}: {corr:.3f}"

plt.figtext(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('btc_forecast_analysis.png', bbox_inches='tight', dpi=300)
plt.show()

# Print detailed analysis
print("\nDetailed Analysis:")
print(metrics_text)

# Additional statistics
print("\nReturn Statistics:")
print(f"Historical Returns (last 30 days):")
print(pd.Series(historical_returns[-30:]).describe())
print("\nForecast Returns (next 30 days):")
print(pd.Series(forecast_returns[:30]).describe())

# Print additional backtesting statistics
print("\nBacktesting Statistics:")
print(f"MAPE: {mape:.2f}%")
print(f"RMSE: ${rmse:.2f}")
print(f"MAE: ${mae:.2f}")
print("\nError Distribution Statistics:")
print(pd.Series(percentage_errors).describe())

# Additional analysis printing
print("\nFeature Statistics:")
for column in market_data.columns:
    print(f"\n{column} Statistics:")
    print(market_data[column].describe())