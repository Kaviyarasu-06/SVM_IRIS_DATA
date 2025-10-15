import streamlit as st
import pickle
import numpy as np

st.title("🌼 Iris Flower Species Prediction using SVM Kernels")
st.write("Select a kernel type and enter flower measurements to predict the species.")

# Dropdown for selecting kernel type
kernel = st.selectbox("Choose SVM Kernel", ["Linear", "Polynomial", "RBF"])

# Input fields for features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

# Predict button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Load selected model
    model_file = {
        "Linear": "svm_linear.pkl",
        "Polynomial": "svm_poly.pkl",
        "RBF": "svm_rbf.pkl"
    }[kernel]

    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    # Make prediction
    prediction = model.predict(input_data)[0]
    st.success(f"🌸 Predicted Iris Species ({kernel} Kernel): **{prediction}**")
