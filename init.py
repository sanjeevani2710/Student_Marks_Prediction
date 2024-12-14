from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open("student_marks_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    # Initialize variables for the form and predictions
    Prediction_text = None
    Error_text = None

    # Variables to retain user input
    study_hours = None
    attendance = None
    past_marks = None
    extracurricular = None

    if request.method == "POST":
        try:
            # Fetch input values from the form
            study_hours = float(request.form.get("Study_Hours", 0))
            attendance = float(request.form.get("Attendance", 0))
            past_marks = float(request.form.get("Past_Marks", 0))
            extracurricular = int(request.form.get("Extracurricular", 0))  # 'Yes' -> 1, 'No' -> 0

            # Validate inputs
            if not (0 <= attendance <= 100):
                raise ValueError("Attendance percentage must be between 0 and 100.")
            if not (0 <= past_marks <= 100):
                raise ValueError("Past marks must be between 0 and 100.")
            if not (0 <= study_hours <= 24):
                raise ValueError("Study hours must be between 0 and 24.")

            # Prepare the input for the model
            input_features = np.array([[study_hours, attendance, past_marks, extracurricular]])
            
            # Make prediction
            predicted_marks = model.predict(input_features)[0]
            Prediction_text = f"Predicted Marks: {predicted_marks:.2f}"

        except ValueError as ve:
            Error_text = str(ve)
        except Exception as e:
            Error_text = "An error occurred during prediction. Please check your inputs."

    return render_template(
        "index.html",
        Prediction_text=Prediction_text,
        Error_text=Error_text,
        study_hours=study_hours,
        attendance=attendance,
        past_marks=past_marks,
        extracurricular=extracurricular
    )

if __name__ == "__main__":
    app.run(debug=True)
