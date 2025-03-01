from flask import Flask, render_template, request
import joblib
import pandas as pd
import logging

# Initialize Flask application
app = Flask(__name__)

# Load pre-trained Random Forest model and encoders
rf_model = joblib.load('random_forest_obesity_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define the columns (features) expected by the model
FEATURES = [
    'Gender', 'Age', 'Height', 'Weight', 'Family History with Overweight',
    'Frequent consumption of high caloric food', 'Frequency of consumption of vegetables',
    'Number of main meals', 'Consumption of food between meals', 'Smoke',
    'Consumption of water daily', 'Calories consumption monitoring', 
    'Physical activity frequency', 'Time using technology devices', 
    'Consumption of alcohol', 'Transportation used'
]

@app.route('/')
def home():
    return render_template('index.html')  # The form will be here

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Log that we received a POST request
            logging.info("Received POST request for prediction")

            # Get the form data
            input_data = [
                request.form.get('gender', ''),
                float(request.form.get('age', 0)),
                float(request.form.get('height', 0)),
                float(request.form.get('weight', 0)),
                request.form.get('family_history', ''),
                request.form.get('caloric_food', ''),
                request.form.get('veggies', ''),
                request.form.get('meals', ''),
                request.form.get('food_between_meals', ''),
                request.form.get('smoke', ''),
                request.form.get('water', ''),
                request.form.get('calories_monitoring', ''),
                request.form.get('activity', ''),
                request.form.get('technology', ''),
                request.form.get('alcohol', ''),
                request.form.get('transport', '')
            ]

            # Log the input data
            logging.info("Input data: %s", input_data)

            # Create a DataFrame for the input
            input_df = pd.DataFrame([input_data], columns=FEATURES)

            # Encode the categorical variables using the saved label encoders
            for column in input_df.columns:
                if column in label_encoders:
                    try:
                        input_df[column] = label_encoders[column].transform([input_df[column][0]])
                    except ValueError:
                        logging.error("Unseen label for column %s: %s", column, input_df[column][0])
                        # You may want to set a default value or handle it accordingly
                        input_df[column] = label_encoders[column].transform(['default_value'])  # Replace 'default_value' with a suitable default

            # Log the encoded input DataFrame
            logging.info("Encoded input data: %s", input_df)

            # Make prediction using the Random Forest model
            rf_prediction = rf_model.predict(input_df)

            # Log the raw prediction result
            logging.info("Raw model prediction: %s", rf_prediction)

            # Decode the prediction
            rf_prediction = label_encoders['Obesity'].inverse_transform(rf_prediction)[0]

            # Log the decoded prediction
            logging.info("Decoded prediction: %s", rf_prediction)

            # Return the result to the result page
            return render_template('result.html', rf_prediction=rf_prediction)

        except Exception as e:
            logging.error("Error during prediction: %s", e)
            # Render home page with an error message if something goes wrong
            return render_template('index.html', error="An error occurred during prediction. Please try again.")

if __name__ == '__main__':
    app.run(debug=True)
