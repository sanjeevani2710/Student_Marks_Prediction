import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from zipfile import ZipFile
import io
import requests

# Download the ZIP file from the URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
response = requests.get(url)

# Open the ZIP file and extract the 'student-mat.csv' file
with ZipFile(io.BytesIO(response.content)) as zip_file:
    zip_file.printdir()  # List all files in the ZIP archive
    
    with zip_file.open('student-mat.csv') as file:
        data = pd.read_csv(file, delimiter=";")

# Map 'yes'/'no' values in the 'activities' column to 1/0
data['activities'] = data['activities'].map({'yes': 1, 'no': 0})

# Select the relevant features for prediction
features = ['studytime', 'absences', 'G1', 'activities']
target = 'G3'  # The target variable (final grade)

# Filter data to include only the required features
data_filtered = data[features + [target]].copy()  # Use copy() to avoid SettingWithCopyWarning
data_filtered.dropna(inplace=True)

# Split the data into features (X) and target (y)
X = data_filtered[features]
y = data_filtered[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert X_train and X_test to numpy arrays without column names to avoid the warning
X_train = X_train.values
X_test = X_test.values

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Clip the predictions to ensure they are within the range [0, 100]
y_pred_clipped = np.clip(y_pred, 0, 100)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred_clipped)
mse = mean_squared_error(y_test, y_pred_clipped)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Save the model using pickle
with open('student_marks_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Example prediction function
def predict_marks(studytime, absences, G1, activities):
    input_data = np.array([[studytime, absences, G1, activities]])
    predicted_marks = model.predict(input_data)
    predicted_marks_clipped = np.clip(predicted_marks, 0, 100)
    return round(predicted_marks_clipped[0], 2)

# Example: Get predicted marks for a new student
studytime = 3  # Example studytime
absences = 5   # Example absences
G1 = 15        # Example past marks
activities = 1 # Example extracurricular activity (1 means yes)

predicted_grade = predict_marks(studytime, absences, G1, activities)
print(f"Predicted Final Grade: {predicted_grade}")
