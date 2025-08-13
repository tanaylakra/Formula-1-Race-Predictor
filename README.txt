# 🏎️ 2024 British Grand Prix – Lap Time Prediction & Analysis (FastF1 + Python)

This project uses the **[FastF1](https://theoehrly.github.io/Fast-F1/)** Python library to analyze **Formula 1 2024 British Grand Prix** data from practice, qualifying, and the race.  
It trains a simple **Linear Regression** model to predict lap times for all drivers based on tyre, stint, and team data.  

The script also fetches **qualifying results** and **final race results**, and visualizes predicted lap times in a bar chart.

---

## 📌 Features
- **Load F1 data** from all practice sessions (FP1, FP2, FP3) for the 2024 British GP.
- **Clean & preprocess data** (driver, team, tyre compound, stint, tyre life, session).
- **Encode categorical variables** for machine learning.
- **Train a Linear Regression model** to predict lap times.
- **Aggregate predictions** for all drivers.
- **Visualize results** in a bar chart.
- **Fetch official qualifying & race results** for comparison.

---

## 🛠️ Requirements

Before running the script, install these Python libraries:

```bash
pip install fastf1 pandas numpy matplotlib scikit-learn
