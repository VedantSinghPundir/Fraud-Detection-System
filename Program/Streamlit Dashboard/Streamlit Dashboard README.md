## README: Streamlit Dashboard Program 

### **Overview**  
The Streamlit dashboard provides an interactive interface to explore and analyze transactional data for fraud detection.

### **Key Functionalities**  
1. **Data Display**:
   - Shows sample transaction data.
   - Provides summaries, such as total transactions and fraud cases.
2. **Visualizations**:
   - Fraud distribution (Histogram or Pie chart).
   - World map of fraudulent transactions by country.
3. **Feature Engineering**:
   - Log transformation of transaction amounts.
   - Encoding categorical data and creating additional features.
4. **Machine Learning**:
   - Prepares data for Random Forest classification.
   - Trains and evaluates the model.

### **Usage Instructions**  
1. Ensure the dataset is available at the specified path in the code.
2. Run the Streamlit application:
   ```bash
   streamlit run fraud_dashboard.py
   ```

### **Core Code Highlights**  
- **Visualization**:  
  Fraud distribution visualized using Plotly's histogram and pie chart.
- **Model Training**:  
  Random Forest Classifier trains on preprocessed features.
- **Evaluation**:  
  Displays classification metrics such as accuracy, confusion matrix, and classification report.

---
