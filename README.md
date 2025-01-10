# Fraud Detection System

## Description
This project is a comprehensive fraud detection system designed to identify fraudulent transactions in a dataset. It combines a user-friendly dashboard for data visualization and analysis with a REST API for predictive modeling. Using machine learning techniques, the system identifies potential fraud and provides interactive insights.

## Key Features
- **Streamlit Dashboard**:
  - Visualize transaction data distribution.
  - Analyze fraudulent transactions by country and type.
  - Interactive world map visualization.
- **Machine Learning**:
  - Random Forest Classifier for fraud detection.
  - Preprocessing with feature engineering.
- **REST API**:
  - Predict fraud probability using Flask.

## Technologies Used
- **Programming Language**: Python  
- **Libraries**: Streamlit, Flask, Scikit-learn, Plotly, Pandas, NumPy  
- **Visualization Tools**: Plotly for interactive charts, Streamlit for dashboards  
- **Model Deployment**: Flask for API

## Installation

### Prerequisites
- Python 3.8 or above
- Required libraries (see `requirements.txt`)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/fraud-detection-system.git
   ```
2. Navigate to the project directory:
   ```bash
   cd fraud-detection-system
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the dataset and place it in the specified path:
   - Update the `data_path` variable in `fraud_dashboard.py` with the correct dataset location.

## Usage

### Streamlit Dashboard
1. Run the Streamlit application:
   ```bash
   streamlit run fraud_dashboard.py
   ```
2. Open the URL provided by Streamlit in your browser (default: `http://localhost:8501`).
3. Explore data visualizations, analyze fraud distribution, and view model results.

### Flask API
1. Start the Flask server:
   ```bash
   python flask_api.py
   ```
### Use the API to make predictions by sending a POST request to /predict.
   

## Dataset
This project uses a transactional dataset that includes:
- **Features**: 
  - `Time_step`: Timestamp of the transaction.
  - `USD_amount`: Transaction amount in USD.
  - `Sender_Account`, `Sender_Country`: Sender details.
  - `Bene_Country`: Beneficiary country.
  - `Transaction_Type`: Type of transaction.
- **Target**: `Label` (1 for fraud, 0 for non-fraud).


## Future Improvements
- Implementing advanced machine learning techniques like ensemble methods or deep learning.
- Adding more robust error handling in the Flask API.
- Including real-time transaction monitoring capabilities.

