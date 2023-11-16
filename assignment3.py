import streamlit as st
import joblib
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense

# Define create_mlp_model function first
def create_mlp_model(input_shape, optimizer='adam'):
    inputs = Input(shape=(input_shape,))
    hidden_layer1 = Dense(64, activation='relu')(inputs)
    hidden_layer2 = Dense(32, activation='relu')(hidden_layer1)
    hidden_layer3 = Dense(16, activation='relu')(hidden_layer2)
    output_layer = Dense(1, activation='sigmoid')(hidden_layer3)

    model = Model(inputs=inputs, outputs=output_layer)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Load the trained model
model_path = 'churn_model.pkl'
with open(model_path, 'rb') as model_file:
    model = joblib.load(model_file)

# Streamlit app setup
st.title("Churn Prediction App")

# Create a form for users to input new data
st.sidebar.header("Enter Customer Information:")
senior_citizen = st.sidebar.number_input("Senior Citizen (0 for No, 1 for Yes)", min_value=0, max_value=1, step=1)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=120, step=1)
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0, step=1)
total_charges = st.sidebar.number_input("Total Charges", min_value=0, step=1)
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])


# Map 'Contract' to numerical values
contract_mapping = {
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2
}
contract_encoded = contract_mapping.get(contract, 0)

internet_service_mapping = {
    'DSL': 0,
    'Fiber optic': 1,
    'No': 2
}
internet_service_encoded = internet_service_mapping.get(internet_service, 0)

binary_mapping = {'Yes': 1, 'No': 0, 'No internet service': 0}
online_security_encoded = binary_mapping[online_security]
tech_support_encoded = binary_mapping[tech_support]

# Convert user input to DataFrame
input_data = pd.DataFrame({
    'SeniorCitizen': [senior_citizen],
    'gender': [1 if gender == "Male" else 0],  # Assuming 1 for Male, 0 for Female
    'Partner': [1 if partner == "Yes" else 0],  # Assuming 1 for Yes, 0 for No
    'Dependents': [1 if dependents == "Yes" else 0],  # Assuming 1 for Yes, 0 for No
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Contract': [contract_encoded],
    'InternetService': [internet_service_encoded ],
    'OnlineSecurity': [online_security_encoded],
    'TechSupport': [tech_support_encoded]
})

if st.button("Predict"):
    # Predict using the loaded model
    prediction = model.predict(input_data)

    # Display the prediction and confidence factor
    st.subheader("Prediction:")
    if prediction.flatten()[0] >= 0.5:
        st.write("The model predicts that the customer is likely to churn.")
    else:
        st.write("The model predicts that the customer is not likely to churn.")

    st.subheader("Confidence Factor:")
    confidence_factor = prediction.flatten()[0]
    st.write(f"The confidence factor of the prediction is: {confidence_factor:.2%}")
