import torch
import pandas as pd
import joblib

from src.model import MLP


# ==========================================
# FEATURE NAMES
# ==========================================

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


# ==========================================
# LOAD PRIVATE GLOBAL MODEL
# ==========================================

model = MLP(input_size=13)

model.load_state_dict(
    torch.load(
        "models/private_global_model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()


# ==========================================
# LOAD TRAINING SCALER
# ==========================================

scaler = joblib.load(
    "models/scaler_A.pkl"
)


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_heart_disease(patient_data):

    # Convert patient data into DataFrame

    df = pd.DataFrame(
        [patient_data],
        columns=FEATURES
    )


    # Convert values to numeric

    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )


    # Fill missing values

    df = df.fillna(0)


    # Use the SAME scaler
    # that was used during training

    X = scaler.transform(df)


    # Convert to PyTorch tensor

    X = torch.tensor(
        X,
        dtype=torch.float32
    )


    # Make prediction

    with torch.no_grad():

        output = model(X)


    probability = output.item()


    # Convert to percentage

    percentage = probability * 100


    # Classification

    if probability >= 0.5:

        prediction = "Positive"

    else:

        prediction = "Negative"


    return prediction, percentage


# ==========================================
# TEST PREDICTION
# ==========================================

if __name__ == "__main__":

    sample_patient = [


    45,
    0,
    1,
    120,
    200,
    0,
    0,
    170,
    0,
    0.5,
    1,
    0,
    2

]


    prediction, probability = predict_heart_disease(
        sample_patient
    )


    print()

    print(
        "Heart Disease Prediction:",
        prediction
    )

    print(
        f"Probability: {probability:.2f}%"
    )