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