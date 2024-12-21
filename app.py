from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Initialize Flask application
app = Flask(__name__)

# Load the trained model and preprocessor pipeline
with open('best_model.pkl', 'rb') as f:
    best_model_pipeline = pickle.load(f)

# Define the route for predicting whether a shipment is on time or delayed
@app.route('/', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Check that all required keys are in the input data
        required_columns = ['Origin', 'Destination', 'Vehicle Type', 'Distance (km)', 'Weather Conditions', 'Traffic Conditions']
        
        # Ensure all required fields are provided
        if not all(col in data for col in required_columns):
            return jsonify({'error': 'Missing required fields'}), 400

        # Convert the incoming data into a format suitable for DataFrame (a dictionary with lists)
        input_data = {col: [data[col]] for col in required_columns}

        # Convert input_data to a DataFrame
        input_df = pd.DataFrame(input_data)

        # Use the model to make predictions
        prediction = best_model_pipeline.predict(input_df)
        
        # Return the prediction result
        result = 'Delayed' if prediction[0] == 1 else 'On Time'
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
