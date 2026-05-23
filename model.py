import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("student_data.csv")

# Inputs
X = data[['StudyHours', 'SleepHours', 'PreviousScore']]

# Output
y = data['FinalScore']

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Save trained model
pickle.dump(model, open('model.pkl', 'wb'))

print("Model trained successfully!")