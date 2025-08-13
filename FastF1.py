import fastf1
from fastf1.core import Laps
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Create and enable cache
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

# Load all practice sessions
sessions = ['FP1', 'FP2', 'FP3']
laps_df = pd.DataFrame()

for session_name in sessions:
    session = fastf1.get_session(2024, 'British Grand Prix', session_name)
    session.load()
    laps = session.laps.pick_accurate().pick_quicklaps()
    laps.loc[:, 'Session'] = session_name
    laps_df = pd.concat([laps_df, laps])

# Ensure required columns exist
if 'Stint' not in laps_df.columns:
    laps_df['Stint'] = 1  # Default if missing

# Select and process columns
laps_df = laps_df[['Driver', 'Team', 'LapTime', 'Compound', 'TyreLife', 'Stint', 'Session']]
laps_df['LapTimeSeconds'] = laps_df['LapTime'].dt.total_seconds()

# Encode categorical data
le_driver = LabelEncoder()
le_team = LabelEncoder()
le_compound = LabelEncoder()
le_session = LabelEncoder()

laps_df['DriverEncoded'] = le_driver.fit_transform(laps_df['Driver'])
laps_df['TeamEncoded'] = le_team.fit_transform(laps_df['Team'])
laps_df['CompoundEncoded'] = le_compound.fit_transform(laps_df['Compound'])
laps_df['SessionEncoded'] = le_session.fit_transform(laps_df['Session'])

# Train model
features = ['TeamEncoded', 'CompoundEncoded', 'TyreLife', 'Stint', 'SessionEncoded']
target = 'LapTimeSeconds'

X = laps_df[features]
y = laps_df[target]

model = LinearRegression()
model.fit(X, y)

# Predict lap times
laps_df['PredictedLapTime'] = model.predict(X)

# Aggregate predictions by driver
driver_predictions = laps_df.groupby('Driver', as_index=False)[['PredictedLapTime']].mean()

# Map driver codes to full names
driver_name_map = {
    'VER': 'Max Verstappen',
    'HAM': 'Lewis Hamilton',
    'NOR': 'Lando Norris',
    'LEC': 'Charles Leclerc',
    'SAI': 'Carlos Sainz',
    'PER': 'Sergio Pérez',
    'ALO': 'Fernando Alonso',
    'RUS': 'George Russell',
    'PIA': 'Oscar Piastri',
    'BOT': 'Valtteri Bottas',
    'ZHO': 'Guanyu Zhou',
    'OCO': 'Esteban Ocon',
    'GAS': 'Pierre Gasly',
    'HUL': 'Nico Hülkenberg',
    'MAG': 'Kevin Magnussen',
    'TSU': 'Yuki Tsunoda',
    'ALB': 'Alex Albon',
    'SAR': 'Logan Sargeant',
    'RIC': 'Daniel Ricciardo',
    'LAW': 'Liam Lawson',
    'DOO': 'Jack Doohan',
    'COL': 'Colton Herta',
    'BEA': 'Oliver Bearman',
    'HAD': 'Isack Hadjar'
}

driver_predictions['FullName'] = driver_predictions['Driver'].map(driver_name_map)

# Sort results
driver_predictions_sorted = driver_predictions.sort_values('PredictedLapTime')

# Display all
print("\n🏁 Predicted Race Pace – All Drivers:\n")
print(driver_predictions_sorted[['FullName', 'PredictedLapTime']])

# Plot full list
plt.figure(figsize=(12, 8))
plt.barh(
    driver_predictions_sorted['FullName'],
    driver_predictions_sorted['PredictedLapTime'],
    color='royalblue'
)
plt.xlabel("Predicted Lap Time (seconds)")
plt.title("🏎️ 2024 British GP: Predicted Race Pace (All Drivers)")
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
