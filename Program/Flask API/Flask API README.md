## README: Flask API Program

### **Overview**  
The Flask-based API enables real-time prediction of fraudulent transactions using the trained Random Forest model.

### **Endpoints**  
- **`/predict`**:  
  - Method: POST  
  - Input: JSON object containing features like `Hour_of_Day`, `Sender_Account_Code`, `Sender_Country_Code`, `USD_amount`, and `USD_amount_avg_sender`.  
  - Output: Fraud prediction (0 or 1).  

### **Usage Instructions**  
1. Ensure the trained model (`fraud_detection_model.pkl`) is in the same directory as the Flask app.
2. Run the Flask application:
   ```bash
   python flask_api.py
   ```
3. Test the API using tools like Postman or cURL:
   ```bash
   curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"Hour_of_Day": 12, "Sender_Account_Code": 1001, "Sender_Country_Code": 34, "USD_amount": 1000, "USD_amount_avg_sender": 500}'
   ```

### **Core Code Highlights**  
- **Model Loading**:  
  Loads the trained model using `joblib`.
- **Input Processing**:  
  Converts input features into the format required by the model.
- **Error Handling**:  
  Returns detailed error messages if the input is invalid.

---
