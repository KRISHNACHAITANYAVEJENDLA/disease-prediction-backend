import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from flask_cors import CORS # Make sure this is imported correctly
import json
import os

app = Flask(__name__)

# --- IMPORTANT CORS SETTING FOR TROUBLESHOOTING ---
# This line enables Cross-Origin Resource Sharing (CORS) for your Flask application.
# Currently, it's set to allow requests from *any* origin. This is a temporary
# setting to help diagnose your "Failed to fetch" error.
#
# Once your frontend is successfully communicating with your backend,
# it's recommended to make this more secure by specifying your exact GitHub Pages URL:
# Example (uncomment and replace the line below 'CORS(app)' with this for production):
CORS(app, origins=["https://KRISHNACHAITANYAVEJENDLA.github.io/disease-prediction"])
#CORS(app) # <-- This line is active and allows all origins for testing.

# --- Load and Prepare Data ---
try:
    # Ensure 'training.csv' is located in the same directory as this app.py file
    # when deployed on Render. If your CSV is elsewhere, adjust the path accordingly.
    df = pd.read_csv('training.csv')

    # Assuming your 'training.csv' has 'prognosis' as the disease column
    # and 'medicine' as the medication column, these are typically the last two columns.
    diseases = df['prognosis']
    medications = df['medicine'] # Storing for the disease-to-medication map

    # All columns except the last two are assumed to be symptom features for the model.
    symptom_columns = df.columns[:-2].tolist()
    X = df[symptom_columns] # Features (symptoms)
    y = diseases # Target variable (disease/prognosis)

    # Initialize and train the Decision Tree Classifier model.
    # This model learns to map symptom patterns to diseases.
    model = DecisionTreeClassifier()
    model.fit(X, y)

    # Create a dictionary to easily look up medication based on a predicted disease.
    disease_med_map = df.set_index('prognosis')['medicine'].to_dict()

    print("Model trained and data loaded successfully!")

except FileNotFoundError:
    # This block executes if 'training.csv' is not found.
    # It's a critical file for the application, so we print an error and exit.
    print("Error: training.csv not found. Make sure it's in the same directory as app.py and your Render build process copies it.")
    exit() # Terminate the application startup

except Exception as e:
    # This catches any other unexpected errors during the data loading or model training phase.
    print(f"An error occurred during data loading or model training: {e}")
    exit() # Terminate the application startup if a critical error occurs

# --- API Endpoint for Disease Prediction ---
# This route handles incoming POST requests to the '/predict_disease' path.
@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    # The frontend sends a JSON payload. We expect it to have a 'symptoms' key
    # containing a list of strings (e.g., {'symptoms': ['fever', 'cough']}).
    data = request.json
    user_symptoms = data.get('symptoms', []) # Safely retrieves 'symptoms' list, defaults to empty list if not found.

    # Basic validation: If no symptoms are provided, return an error.
    if not user_symptoms:
        return jsonify({'error': 'No symptoms provided.'}), 400 # HTTP 400 Bad Request

    # Create a DataFrame row for the input to the model.
    # It's initialized with zeros for all possible symptom columns.
    input_symptom_vector = pd.DataFrame(0, index=[0], columns=symptom_columns)

    # Process each symptom provided by the user.
    for symptom in user_symptoms:
        # Normalize the symptom text to match the format used in 'training.csv' column names.
        # This typically means converting to lowercase and replacing spaces with underscores.
        normalized_symptom = symptom.strip().replace(' ', '_').lower()

        # Check if the normalized symptom exists as a column in our training data.
        if normalized_symptom in input_symptom_vector.columns:
            # If it exists, mark it as present (1) in our input vector.
            input_symptom_vector[normalized_symptom] = 1
        else:
            # If the symptom is not recognized (i.e., not in our training data),
            # print a warning to the console but continue processing.
            print(f"Warning: Symptom '{symptom}' (normalized: '{normalized_symptom}') not found in training data. Ignoring.")

    # Use the pre-trained Decision Tree model to predict the disease based on the input symptoms.
    predicted_disease = model.predict(input_symptom_vector)[0] # [0] to get the first prediction

    # Retrieve the corresponding medication for the predicted disease from our map.
    # If a medication isn't found for some reason, a default message is used.
    predicted_medication = disease_med_map.get(predicted_disease, "No specific medication found for this disease.")

    # Return the predicted disease and medication as a JSON response to the frontend.
    return jsonify({
        'disease': predicted_disease,
        'medication': predicted_medication
    })

# --- Main entry point for the Flask application when run directly ---
if __name__ == '__main__':
    # Render (and other cloud platforms) typically provides the port number
    # your application should listen on via an environment variable named 'PORT'.
    # We retrieve it using os.environ.get("PORT", 5000), defaulting to 5000 for local development.
    port = int(os.environ.get("PORT", 5000))

    # Run the Flask application:
    # host='0.0.0.0' makes the server accessible from all network interfaces,
    # which is crucial for deployments on platforms like Render.
    # debug=True enables Flask's debugger (useful during development, but set to False for production).
    app.run(debug=True, host='0.0.0.0', port=port)
