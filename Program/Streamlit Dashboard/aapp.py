import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import joblib

st.title("Fraud Detection Dashboard")


data_path = "C:/Users/singh/OneDrive/Desktop/Vedant/Combined College Files/COLLEGE/Sem 6 files/sem 6 min projrct/new_data.csv"  

@st.cache_data(persist=True)
def load_data():
    data = pd.read_csv(data_path)
    return data

data = load_data()


st.subheader("Sample Data")
st.write(data.head())


st.sidebar.subheader("Data Summary")
st.sidebar.markdown(f"Total transactions: {len(data)}")


fraud_column = 'Label' if 'Label' in data.columns else None
if fraud_column:
    st.sidebar.markdown(f"Total fraud cases: {data[fraud_column].sum()}")

    st.sidebar.subheader("Visualization")
    select = st.sidebar.selectbox('Visualization type', ['Histogram', 'Pie chart'], key='vis_type')
    fraud_count = data[fraud_column].value_counts().reset_index()
    fraud_count.columns = ['IsFraud', 'Count']

    if not st.sidebar.checkbox("Hide", True):
        st.markdown("### Fraud Distribution")
        if select == "Histogram":
            fig = px.bar(fraud_count, x='IsFraud', y='Count', color='Count', height=500)
            st.plotly_chart(fig)
        else:
            fig = px.pie(fraud_count, values='Count', names='IsFraud')
            st.plotly_chart(fig)

    st.sidebar.subheader("Transaction Analysis")

    st.sidebar.subheader("Fraudulent Transactions")
    fraud_data = data[data[fraud_column] == 1]

    if not st.sidebar.checkbox("Hide", True, key='hide_fraud_transactions'):
        st.markdown("### Details of Fraudulent Transactions")
        st.write(fraud_data)

    
    bene_country_column = 'Bene Country' if 'Bene Country' in data.columns else 'Bene_Country'
    if bene_country_column in data.columns:
        st.sidebar.subheader("Analysis by Bene Country")
        selected_country = st.sidebar.selectbox('Select Bene Country', data[bene_country_column].unique(), key='bene_country')

        if not st.sidebar.checkbox("Hide", True, key='hide_bene_country_analysis'):
            country_data = data[data[bene_country_column] == selected_country]
            st.markdown(f"### Transactions for {selected_country}")
            st.write(country_data)

            country_fraud_count = country_data[fraud_column].value_counts().reset_index()
            country_fraud_count.columns = ['IsFraud', 'Count']

            if not st.sidebar.checkbox("Hide Country Fraud Distribution", True, key='hide_country_fraud_distribution'):
                st.markdown(f"### Fraud Distribution for {selected_country}")
                fig_country = px.bar(country_fraud_count, x='IsFraud', y='Count', color='Count', height=500)
                st.plotly_chart(fig_country)


st.subheader("World Map Visualization")


fraud_data = data[data['Label'] == 1]
transaction_counts = fraud_data['Bene_Country'].value_counts().reset_index()
transaction_counts.columns = ['Country', 'Count']


transaction_counts['LogCount'] = np.log1p(transaction_counts['Count'])


fig = px.choropleth(transaction_counts,
                    locations="Country",
                    locationmode='country names',
                    color="LogCount",
                    hover_name="Country",
                    hover_data={'Country': True, 'Count': True, 'LogCount': False},  
                    color_continuous_scale=px.colors.sequential.Plasma,
                    title="Number of Transactions per Country")


fig.update_layout(
    title={
        'text': "Number of Transactions per Country",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    geo=dict(
        showcoastlines=True,
        coastlinecolor="Black",
        showland=True,
        landcolor="LightGray",
        showocean=True,
        oceancolor="LightBlue",
        projection_type='equirectangular'
    ),
    coloraxis_colorbar=dict(
        title="Log of Number of Transactions",
        ticks="outside"
    )
)

st.plotly_chart(fig)

st.subheader("Preprocessing for Machine Learning")

# feature engineering
data['Hour_of_Day'] = pd.to_datetime(data['Time_step']).dt.hour
data['Log_Transaction_Amount'] = np.log1p(data['USD_amount'])
label_encoder = LabelEncoder()
data['Sender_Country_Code'] = label_encoder.fit_transform(data['Sender_Country'])
data = pd.get_dummies(data, columns=['Transaction_Type'])
average_amount_per_sender = data.groupby('Sender_Account')['USD_amount'].mean()
data = data.merge(average_amount_per_sender, on='Sender_Account', how='left', suffixes=('', '_avg_sender'))
data['Sender_Account_Code'] = label_encoder.fit_transform(data['Sender_Account'])

st.write("Data after feature engineering:")
st.write(data.head())

st.sidebar.header("Machine Learning Model")

st.subheader("Preprocessing for Machine Learning")
st.write("Performing data preprocessing...")
X = data[['Hour_of_Day', 'Sender_Account_Code', 'Sender_Country_Code', 'Log_Transaction_Amount', 'USD_amount_avg_sender']]
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'fraud_detection_model.pkl')


accuracy = model.score(X_test, y_test)

st.title('Fraud Detection Model Evaluation')
st.write(f"Accuracy: {accuracy}")

st.subheader("Model Evaluation")
y_pred = model.predict(X_test)
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

st.write("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
st.write(cm)