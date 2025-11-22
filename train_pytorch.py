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

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

class RegresiNN(nn.Module):
    def __init__(self, input_size):
        super(RegresiNN, self).__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.relu = torch.relu()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x



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

X_train_tensor = torch.from_numpy(X_train_scaled).float()
y_train_tensor = torch.from_numpy(y_train.values).float().unsqueeze(1) # unsqueeze(1) makes it a (N, 1) shape

X_test_tensor = torch.from_numpy(X_test_scaled).float()
y_test_tensor = torch.from_numpy(y_test.values).float().unsqueeze(1)

input_size = X_train_tensor.shape[1]

model = RegressionNN(input_size)
# Mean Squared Error is the standard loss function for regression
criterion = nn.MSELoss() 
# Adam is a popular, effective optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001) 
epochs = 500

for epoch in range(epochs):
    # Set model to training mode
    model.train() 
    
    # Forward pass: compute predicted y by passing x to the model
    y_pred = model(X_train_tensor)
    
    # Compute loss
    loss = criterion(y_pred, y_train_tensor)
    
    # Zero the gradients (clear gradients from previous iteration)
    optimizer.zero_grad()
    
    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    
    # Update weights/parameters
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

r_squared = 0.0

model.eval() 
with torch.no_grad(): # Disable gradient calculations for efficiency
    y_predicted_tensor = model(X_test_tensor)
    
    # Convert PyTorch Tensor predictions back to a 1D NumPy array
    y_predicted_np = y_predicted_tensor.squeeze().numpy()
    y_test_np = y_test_tensor.squeeze().numpy()
    
    # Calculate R-squared using the NumPy arrays
    r_squared = r2_score(y_test_np, y_predicted_np)

print("---")
print(f"Model: PyTorch Neural Network Regression")
print(f"The calculated R-squared score is: {r_squared:.4f}")

print(f"The calculated R-squared score is: {r_squared:.4f}")

