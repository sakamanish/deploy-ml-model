
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Train predictive model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X, y)

# Step 3: Create web interface
st.title("Iris Flower Prediction")
st.write("Enter flower measurements to predict its species.")

sepal_length = st.number_input(
    "Sepal length (cm)", 0.0, 10.0, 5.1
)
sepal_width = st.number_input(
    "Sepal width (cm)", 0.0, 10.0, 3.5
)
petal_length = st.number_input(
    "Petal length (cm)", 0.0, 10.0, 1.4
)
petal_width = st.number_input(
    "Petal width (cm)", 0.0, 10.0, 0.2
)

# Step 4: Predict
if st.button("Predict Flower"):
    sample = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(sample)[0]
    species = iris.target_names[prediction]

    st.success(f"Predicted species: {species}")
