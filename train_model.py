import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('Crop.csv')

# Prepare features and target
X = df.drop('label', axis=1)  # Replace 'label' with your target column name
y = df['label']  # Replace 'label' with your target column name

# Encode target if it's text
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('crop_model.pkl', 'wb'))

# Save label encoder for predictions
pickle.dump(le, open('label_encoder.pkl', 'wb'))

print("Model created and saved successfully!")