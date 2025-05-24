import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os

app = Flask(__name__)

CORS(app)

try:
    df = pd.read_csv('training.csv')

    diseases = df['prognosis']
    medications = df['medicine']

    symptom_columns = df.columns[:-2].tolist()
    X = df[symptom_columns]
    y = diseases

    model = DecisionTreeClassifier()
    model.fit(X, y)

    disease_med_map = df.set_index('prognosis')['medicine'].to_dict()

    print("Model trained and data loaded successfully!")

except FileNotFoundError:
    print("Error: training.csv not found. Make sure it's in the same directory as app.py and your Render build process copies it.")
    exit()

except Exception as e:
    print(f"An error occurred during data loading or model training: {e}")
    exit()

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    data = request.json
    user_symptoms = data.get('symptoms', [])

    if not user_symptoms:
        return jsonify({'error': 'No symptoms provided.'}), 400

    input_symptom_vector = pd.DataFrame(0, index=[0], columns=symptom_columns)

    for symptom in user_symptoms:
        normalized_symptom = symptom.strip().replace(' ', '_').lower()
        if normalized_symptom in input_symptom_vector.columns:
            input_symptom_vector[normalized_symptom] = 1
        else:
            print(f"Warning: Symptom '{symptom}' (normalized: '{normalized_symptom}') not found in training data. Ignoring.")

    predicted_disease = model.predict(input_symptom_vector)[0]

    predicted_medication = disease_med_map.get(predicted_disease, "No specific medication found for this disease.")

    return jsonify({
        'disease': predicted_disease,
        'medication': predicted_medication
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
