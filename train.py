# ============================================================
# GRACE vs GLDAS EWH Time Series (2005–2025)
# Full Script from Reading Data to Plotting (Interval 2 Tahun)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.linear_model import LinearRegression # This is now unused, but shown for comparison
from xgboost import XGBRegressor # Use this for regression tasks


colors_rdylbu = {
    'JPL': '#74ADD1',
    'CSR': '#A50026',
    'GSFC': '#FDAE61'
}
# ============================================================
# 1. LOAD DATA GRACE (0.5°)
# ============================================================

grace = pd.read_csv("Input_R_GRACE-GLDAS_2005-2025_0_5.csv",
                    delimiter=',')

grace['time'] = pd.to_datetime(grace['time'])

# GRACE scaled_lwe_cm = TWSA = EWH (cm)
# (SUDAH DALAM CM → tidak perlu dikonversi)
grace['GRACE_EWH_cm'] = grace['scaled_lwe_cm']

grace_ts = (
    grace.groupby("time")['GRACE_EWH_cm']
    .mean()
    .reset_index()
)
combined = grace_ts[
    (grace_ts["time"] >= "2005-01-01") &
    (grace_ts["time"] <= "2025-06-30")
]
# grace['date'] = pd.to_datetime(grace['time'])
# grace.reset_index(inplace=True)
# grace.rename(columns={'index': 'Record_Index'}, inplace=True)

# X = combined['time']
# y = combined['GRACE_EWH_cm']

combined.reset_index(inplace=True)
combined.rename(columns={'index': 'Record_Index'}, inplace=True)

y = combined['GRACE_EWH_cm']
X = combined[['Record_Index']]
# X = grace[['Record_Index']]
# y = grace['scaled_lwe_cm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Linear Regression model
model = XGBRegressor(
    objective='reg:squarederror', # Standard objective for regression
    n_estimators=100, 
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_predicted = model.predict(X_test_scaled)

# Calculate R-squared
r_squared = r2_score(y_test, y_predicted)

print(f"The calculated R-squared score is: {r_squared:.4f}")
# print(combined.head())
# plt.figure(figsize=(16,6))
# plt.plot(X, y,
#          label='GRACE JPL',
#          color=colors_rdylbu['CSR'],
#          linewidth=2,
#          marker='o',
#          markersize=4,
#          markerfacecolor=colors_rdylbu['CSR'],
#          markeredgecolor=colors_rdylbu['CSR'],
#          linestyle='-')

# plt.show()