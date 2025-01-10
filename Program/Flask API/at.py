from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('fraud_detection_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = request.get_json(force=True)
        
        features = [
            data['Hour_of_Day'], data['Sender_Account_Code'], data['Sender_Country_Code'], 
            np.log1p(data['USD_amount']), data['USD_amount_avg_sender']
        ]
        
        features_array = np.array(features).reshape(1, -1)
        
        prediction = model.predict(features_array)
        
        return jsonify({'prediction': str(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
