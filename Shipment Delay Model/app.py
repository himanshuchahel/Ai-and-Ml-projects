from flask import Flask, request, jsonify
import pandas as pd
import pickle
app = Flask(__name__)

with open('best_model.pkl', 'rb') as f:
    best_model_pipeline = pickle.load(f)
@app.route('/', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        required_columns = ['Origin', 'Destination', 'Vehicle Type', 'Distance (km)', 'Weather Conditions', 'Traffic Conditions']
        if not all(col in data for col in required_columns):
            return jsonify({'error': 'Missing required fields'}), 400
        input_data = {col: [data[col]] for col in required_columns}
        input_df = pd.DataFrame(input_data)
        prediction = best_model_pipeline.predict(input_df)
        result = 'Delayed' if prediction[0] == 1 else 'On Time'
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)
