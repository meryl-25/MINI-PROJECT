from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Load pre-trained Random Forest model and encoders
rf_model = joblib.load('random_forest_obesity_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define the columns (features) expected by the model
FEATURES = ['Gender', 'Age', 'Height', 'Weight', 'Family History with Overweight',
            'Frequent consumption of high caloric food', 'Frequency of consumption of vegetables',
            'Number of main meals', 'Consumption of food between meals', 'Smoke',
            'Consumption of water daily', 'Calories consumption monitoring', 'Physical activity frequency',
            'Time using technology devices', 'Consumption of alcohol', 'Transportation used']

@app.route('/')
def home():
    return render_template('index.html')  # The form will be here

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the form data
            input_data = [
                request.form['gender'],  # Ensure this matches the form exactly
                float(request.form['age']),
                float(request.form['height']),
                float(request.form['weight']),
                request.form['family_history'],
                request.form['caloric_food'],
                request.form['veggies'],
                request.form['meals'],
                request.form['food_between_meals'],
                request.form['smoke'],
                request.form['water'],
                request.form['calories_monitoring'],  # Fixed: add this line
                request.form['activity'],
                request.form['technology'],
                request.form['alcohol'],
                request.form['transport']
            ]

            # Create a DataFrame for the input
            input_df = pd.DataFrame([input_data], columns=FEATURES)

            # Encode the categorical variables using the saved label encoders
            for column in input_df.columns:
                if column in label_encoders:
                    input_df[column] = label_encoders[column].transform([input_df[column][0]])

            # Make prediction using the Random Forest model
            rf_prediction = rf_model.predict(input_df)

            # Decode the prediction
            rf_prediction = label_encoders['Obesity'].inverse_transform(rf_prediction)[0]

            # Return the result to the result page
            return render_template('result.html', rf_prediction=rf_prediction)

        except Exception as e:
            # Handle any errors during prediction and show a friendly message
            return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
