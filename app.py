import streamlit as st
import pandas as pd
import pickle
# Load and cache your dataset
@st.cache_data
def load_dataset():
    # Replace 'your_dataset.csv' with your actual CSV file name
    df = pd.read_csv('Crop.csv')  # Change this to your CSV file name
    return df
# Load your trained model (adjust path as needed)
@st.cache_data
def load_model():
    # Replace with your actual model loading code
    return pickle.load(open('crop_model.pkl', 'rb'))
# Load the dataset
df = load_dataset()
model = load_model()
# Streamlit app
st.title("🌾 Crop Prediction System")
st.write("Enter the farming conditions to predict the best crop")
# Optional: Show dataset info
with st.sidebar:
    st.subheader("Dataset Information")
    st.write(f"Total samples: {len(df)}")
    st.write(f"Features: {list(df.columns)}")
    if st.checkbox("Show sample data"):
        st.dataframe(df.head())
# Create input fields for features
st.subheader("Enter Soil and Climate Conditions")
col1, col2 = st.columns(2)
with col1:
    nitrogen = st.number_input("Nitrogen (N)", min_value=0, max_value=300, value=50)
    phosphorus = st.number_input("Phosphorus (P)", min_value=0, max_value=300, value=50)
    potassium = st.number_input("Potassium (K)", min_value=0, max_value=300, value=50)
with col2:
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    ph = st.slider("pH Level", 0.0, 14.0, 7.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
# Prediction
if st.button("🔮 Predict Best Crop", type="primary"):
    # Create input array for prediction
    features = [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]]
    # Convert to DataFrame with correct column names (adjust based on your dataset)
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    input_df = pd.DataFrame(features, columns=feature_names)
    # Make prediction
    try:
        prediction = model.predict(input_df)
        # Display result
        st.success(f"🌱 Recommended Crop: **{prediction[0]}**")
        # Optional: Show prediction confidence if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df)
            confidence = max(proba[0]) * 100
            st.info(f"Prediction Confidence: {confidence:.1f}%")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please check that your model file and input features are compatible")
# Additional information
st.markdown("---")
st.subheader("About this System")
st.write("This system uses machine learning to recommend the best crop based on soil nutrients and climate conditions.")