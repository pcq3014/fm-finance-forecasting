import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set paths
summary_path = r'D:\fm-forecasting\data\results\arima_summary.csv'
results_path = r'D:\fm-forecasting\data\results\arima_results.csv'
artifact_dir = r'C:\Users\admin\.gemini\antigravity\brain\88be54a2-a256-40af-91c3-a52975c42dc9'

# Ensure the artifact directory exists
os.makedirs(artifact_dir, exist_ok=True)

# 1. Plot Metrics
df_summary = pd.read_csv(summary_path)
df_metrics = df_summary[df_summary['ticker'] != 'OVERALL']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('ARIMA Metrics by Ticker', fontsize=16)

sns.barplot(data=df_metrics, x='ticker', y='MAE', ax=axes[0, 0])
axes[0, 0].set_title('Mean Absolute Error (MAE)')
axes[0, 0].tick_params(axis='x', rotation=45)

sns.barplot(data=df_metrics, x='ticker', y='RMSE', ax=axes[0, 1])
axes[0, 1].set_title('Root Mean Square Error (RMSE)')
axes[0, 1].tick_params(axis='x', rotation=45)

sns.barplot(data=df_metrics, x='ticker', y='DA', ax=axes[1, 0])
axes[1, 0].set_title('Directional Accuracy (DA)')
axes[1, 0].tick_params(axis='x', rotation=45)

sns.barplot(data=df_metrics, x='ticker', y='IC', ax=axes[1, 1])
axes[1, 1].set_title('Information Coefficient (IC)')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
metrics_plot_path = os.path.join(artifact_dir, 'arima_metrics.png')
plt.savefig(metrics_plot_path)
plt.close()
print(f"Saved {metrics_plot_path}")

# 2. Plot Predictions vs Actuals
df_results = pd.read_csv(results_path)
df_results['Date'] = pd.to_datetime(df_results['Date'])

tickers = df_results['ticker'].unique()
n_tickers = len(tickers)
cols = 3
rows = (n_tickers + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
fig.suptitle('ARIMA Predictions vs Actual Prices', fontsize=20, y=1.02)
axes = axes.flatten()

for i, ticker in enumerate(tickers):
    ax = axes[i]
    df_ticker = df_results[df_results['ticker'] == ticker].sort_values('Date')
    ax.plot(df_ticker['Date'], df_ticker['target'], label='Actual', color='blue', linewidth=1.5)
    ax.plot(df_ticker['Date'], df_ticker['pred'], label='Predicted', color='orange', alpha=0.8, linewidth=1.5)
    ax.set_title(f'{ticker}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)

# Hide any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
preds_plot_path = os.path.join(artifact_dir, 'arima_predictions.png')
plt.savefig(preds_plot_path)
plt.close()
print(f"Saved {preds_plot_path}")
