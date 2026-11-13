# Task 35: Regression Model Evaluation

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create a salary dataset based on experience
data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Salary": [20000, 25000, 30000, 35000, 40000,
               45000, 50000, 55000, 60000, 65000]
}

df = pd.DataFrame(data)

print("Salary Dataset:")
print(df)

# Select input and output columns
X = df[["Experience"]]
y = df["Salary"]

# Divide data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the regression model
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

# Predict salaries using test data
predicted_salary = regression_model.predict(X_test)

# Compare actual and predicted salaries
comparison = pd.DataFrame({
    "Actual Salary": y_test.values,
    "Predicted Salary": predicted_salary
})

print("\nActual Salary vs Predicted Salary:")
print(comparison.round(2))

# Calculate model performance
mae = mean_absolute_error(y_test, predicted_salary)
mse = mean_squared_error(y_test, predicted_salary)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predicted_salary)

# Display evaluation results
print("\nModel Evaluation Results:")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Plot actual and predicted salary values
plt.scatter(X_test, y_test, label="Actual Salary")
plt.scatter(X_test, predicted_salary, label="Predicted Salary")

plt.xlabel("Experience (Years)")
plt.ylabel("Salary (Rs.)")
plt.title("Actual vs Predicted Salary")

plt.legend()
plt.grid(True)
plt.show()
