import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =========================
# 📥 Load Dataset (LOCAL)
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df

df = load_data()

# =========================
# 🧠 Train Model
# =========================
@st.cache_resource
def train_model(df):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, scaler

model, scaler = train_model(df)

# =========================
# 🎯 UI Section
# =========================
def predict():
    st.title("🩺 Diabetes Prediction App")
    st.write("Enter patient details below:")

    name = st.text_input("Name")

    col1, col2 = st.columns(2)

    with col1:
        pregnancy = st.number_input("Pregnancies", min_value=0)
        glucose = st.number_input("Glucose", min_value=0)
        bp = st.number_input("Blood Pressure", min_value=0)
        skin = st.number_input("Skin Thickness", min_value=0)

    with col2:
        insulin = st.number_input("Insulin", min_value=0)
        bmi = st.number_input("BMI", min_value=0.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
        age = st.number_input("Age", min_value=1)

    if st.button("Predict"):
        input_data = np.array([[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if prediction == 0:
            st.success(f"✅ {name}, you are NOT diabetic")
        else:
            st.error(f"⚠️ {name}, you may be diabetic")

        st.write(f"Confidence: {prob*100:.2f}%")

# =========================
# 🏠 Main App
# =========================
def main():
    st.sidebar.title("Menu")
    choice = st.sidebar.selectbox("Select Option", ["Home", "Predict"])

    if choice == "Home":
        st.title("Welcome to Diabetes Prediction App")
        st.write("This app uses Machine Learning to predict diabetes.")
    else:
        predict()

if __name__ == "__main__":
    main()