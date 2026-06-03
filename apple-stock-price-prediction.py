import pandas as pd

# Load the historical quotes CSV file
df = pd.read_csv('HistoricalQuotes.csv')
print(df.head())

# ── Phase 2: Preprocessing ───────────────────
# Fix column name spaces
df.columns = df.columns.str.strip()

# Remove $ signs and convert to numeric
for col in ['Close/Last', 'Open', 'High', 'Low']:
    df[col] = df[col].str.replace('$', '', regex=False).astype(float)

df['Volume'] = df['Volume'].astype(float)

# Convert Date and sort oldest → newest
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Check missing values
print("Missing values:", df.isnull().sum().sum())
print("Shape:", df.shape)
print(df.dtypes)

from sklearn.preprocessing import MinMaxScaler

# Target → next day closing price
df['Target'] = df['Close/Last'].shift(-1)
df = df.dropna().reset_index(drop=True)

# Normalize features
scaler = MinMaxScaler()
features = ['Close/Last', 'Open', 'High', 'Low', 'Volume']
df[features] = scaler.fit_transform(df[features])

print(f"Clean data shape: {df.shape}")
print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

# ── Phase 3: EDA Visualizations ──────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('Apple Stock — Exploratory Analysis',
             fontsize=16, fontweight='bold')

# Chart 1 — Closing Price Over Time
axes[0,0].plot(df['Date'], df['Target'], color='#2E86AB', linewidth=1)
axes[0,0].set_title('Closing Price Over Time', fontweight='bold')
axes[0,0].set_xlabel('Date'); axes[0,0].set_ylabel('Price ($)')
axes[0,0].spines[['top','right']].set_visible(False)

# Chart 2 — Volume Over Time
axes[0,1].bar(df['Date'], df['Volume'], color='#A23B72', alpha=0.6, width=1)
axes[0,1].set_title('Trading Volume Over Time', fontweight='bold')
axes[0,1].set_xlabel('Date'); axes[0,1].set_ylabel('Volume (Normalized)')
axes[0,1].spines[['top','right']].set_visible(False)

# Chart 3 — Moving Averages
axes[1,0].plot(df['Date'], df['Target'], color='#AAAAAA', linewidth=0.8, label='Price')
axes[1,0].plot(df['Date'], df['Target'].rolling(7).mean(),  color='#2ECC71', linewidth=1.5, label='7D MA')
axes[1,0].plot(df['Date'], df['Target'].rolling(30).mean(), color='#E74C3C', linewidth=1.5, label='30D MA')
axes[1,0].set_title('Moving Averages', fontweight='bold')
axes[1,0].set_xlabel('Date'); axes[1,0].set_ylabel('Price ($)')
axes[1,0].legend(fontsize=9)
axes[1,0].spines[['top','right']].set_visible(False)

# Chart 4 — Correlation Heatmap
corr = df[['Close/Last','Open','High','Low','Volume']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[1,1], cbar=True,
            annot_kws={'size':10})
axes[1,1].set_title('Feature Correlation', fontweight='bold')

fig.text(0.5, -0.02, 'Created by Shad Ali Shah',
         color='#555555', fontsize=11, ha='center',
         style='italic', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4',
                   facecolor='#F0F0F0', edgecolor='#CCCCCC'))

plt.tight_layout()
plt.savefig('stock_eda.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

# ── Phase 4: Train/Test Split ─────────────────
features = ['Close/Last', 'Open', 'High', 'Low', 'Volume']
X = df[features].values
y = df['Target'].values

# 80/20 split — NO shuffling (time series order must stay)
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Total  : {len(X)} days")
print(f"Train  : {len(X_train)} days (80%)")
print(f"Test   : {len(X_test)} days (20%)")

# ── Phase 5: Linear Regression ───────────────
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mae_lr  = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr   = r2_score(y_test, y_pred_lr)

print("=" * 40)
print("   LINEAR REGRESSION RESULTS")
print("=" * 40)
print(f"  MAE  : ${mae_lr:.2f}")
print(f"  RMSE : ${rmse_lr:.2f}")
print(f"  R²   : {r2_lr:.4f}")
print("=" * 40)

# ── Phase 6: LSTM ─────────────────────────────
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# Reshape for LSTM → (samples, timesteps, features)
X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_lstm  = X_test.reshape(X_test.shape[0],  1, X_test.shape[1])

# Build LSTM
model = Sequential([
    Input(shape=(1, X_train.shape[1])),
    LSTM(50, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# EarlyStopping → stops when model stops improving
early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)
model.fit(X_train_lstm, y_train, epochs=200, batch_size=32,
          callbacks=[early_stop], verbose=0)

# Evaluate
y_pred_lstm = model.predict(X_test_lstm).flatten()

mae_lstm  = mean_absolute_error(y_test, y_pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
r2_lstm   = r2_score(y_test, y_pred_lstm)

print("=" * 40)
print("       LSTM RESULTS")
print("=" * 40)
print(f"  MAE  : ${mae_lstm:.2f}")
print(f"  RMSE : ${rmse_lstm:.2f}")
print(f"  R²   : {r2_lstm:.4f}")
print("=" * 40)

# ── Future Price Prediction (Linear Regression) ──
future_days = 30
last_data = X_test[-future_days:]

future_pred = lr.predict(last_data)
future_dates = pd.date_range(
    start=df['Date'].iloc[-1], periods=future_days+1, freq='B')[1:]

# Print results
print("=" * 35)
print("  FUTURE STOCK PRICE PREDICTIONS")
print("=" * 35)
for date, price in zip(future_dates, future_pred):
    print(f"  {date.strftime('%Y-%m-%d')}  →  ${price:.2f}")
print("=" * 35)

# ── Plot Future Predictions ───────────────────
fig, ax = plt.subplots(figsize=(13, 5))

ax.plot(df['Date'].iloc[-100:], df['Target'].iloc[-100:],
        color='#2E86AB', linewidth=2, label='Historical Price')
ax.plot(future_dates, future_pred,
        color='#E74C3C', linewidth=2,
        linestyle='--', label='Predicted Future')

ax.axvline(df['Date'].iloc[-1], color='gray',
           linestyle=':', linewidth=1.5, label='Prediction Start')
ax.set_title('Apple Stock — Future Price Prediction',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Date'); ax.set_ylabel('Price ($)')
ax.legend(fontsize=11)
ax.spines[['top','right']].set_visible(False)

fig.text(0.5, -0.02, 'Created by Shad Ali Shah',
         color='#555555', fontsize=11, ha='center',
         style='italic', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4',
                   facecolor='#F0F0F0', edgecolor='#CCCCCC'))

plt.tight_layout()
plt.savefig('future_prediction.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()
