import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error


print("Script started")
# Load data
df = pd.read_csv("expenses.csv")
df["date"] = pd.to_datetime(df["date"])

# AI to Understand 
df["day"] = (df["date"] - df["date"].min()).dt.days
df["day_of_month"] = df["date"].dt.day
df["month"] = df["date"].dt.month



print(df)

# Plot spending
plt.plot(df["date"], df["amount"])
plt.title("Monthly Expenses")
plt.xlabel("Date")
plt.ylabel("Amount")
plt.show()
print("Script Ended")


# Train model
X = df[["day"]]
y = df["amount"]

model = LinearRegression()
model.fit(X, y)

print("Model trained")

# Predict
future_day = np.array([[15]])
prediction = model.predict(future_day)

print("Predicted expense:", prediction[0])
# Predict expense for day 15
future_day = np.array([[15]])
predicted_expense = model.predict(future_day)
print(f"Predicted expense for day 15: {predicted_expense[0]:.2f}")

predictions = model.predict(X)
error = mean_absolute_error(y, predictions)

print("Average prediction error:", error)