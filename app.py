from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    session
)

import torch
import os
import re
import json
import uuid
import webbrowser
import threading

from datetime import datetime
from functools import wraps

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.model import MLP


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "fedhealth-xai-demo-secret-key-change-later"
)


# ============================================================
# DEMO LOGIN USERS
# ============================================================

USERS = {

    "admin": {
        "password": "admin123",
        "role": "admin",
        "hospital": None
    },

    "hospital_a": {
        "password": "hospitalA123",
        "role": "hospital",
        "hospital": "Hospital A"
    },

    "hospital_b": {
        "password": "hospitalB123",
        "role": "hospital",
        "hospital": "Hospital B"
    },

    "hospital_c": {
        "password": "hospitalC123",
        "role": "hospital",
        "hospital": "Hospital C"
    }
}


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "prediction_results.json"
)

RESULTS_LOCK = threading.Lock()


MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)


PRIVATE_GLOBAL_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "private_global_model.pth"
)


LEGACY_GLOBAL_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "global_model.pth"
)


# ============================================================
# PROPOSED ADAPTIVE FEDERATED MODEL
# ============================================================

PROPOSED_ADAPTIVE_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "proposed_adaptive_model.pth"
)


PROPOSED_WEIGHTS_PATH = os.path.join(
    MODELS_DIR,
    "proposed_adaptive_weights.txt"
)


# ============================================================
# LEGACY ADAPTIVE MODEL PATHS
# ============================================================

ADAPTIVE_GLOBAL_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "adaptive_global_model.pth"
)


WEIGHTS_PATH = os.path.join(
    MODELS_DIR,
    "adaptive_weights.txt"
)


SCALER_PATH = os.path.join(
    MODELS_DIR,
    "scaler_A.pkl"
)
# ============================================================
# PROJECT SETTINGS
# ============================================================

TRAINING_ROUNDS = 5

INPUT_FEATURE_COUNT = 13

AGGREGATION_METHOD = (
    "Adaptive Personalized Aggregation"
)

AGGREGATION_ALGORITHM = (
    "Adaptive Weighted Federated Averaging"
)


# ============================================================
# HOSPITAL INFORMATION
# ============================================================

HOSPITALS = {

    "Hospital A": {

        "dataset": os.path.join(
            BASE_DIR,
            "data",
            "hospital_A.csv"
        ),

        "model": os.path.join(
            MODELS_DIR,
            "hospital_A_model.pth"
        ),

        "personalized_model": os.path.join(
            MODELS_DIR,
            "personalized_hospital_A.pth"
        ),

        "scaler": os.path.join(
            MODELS_DIR,
            "scaler_A.pkl"
        )
    },

    "Hospital B": {

        "dataset": os.path.join(
            BASE_DIR,
            "data",
            "hospital_B.csv"
        ),

        "model": os.path.join(
            MODELS_DIR,
            "hospital_B_model.pth"
        ),

        "personalized_model": os.path.join(
            MODELS_DIR,
            "personalized_hospital_B.pth"
        ),

        "scaler": os.path.join(
            MODELS_DIR,
            "scaler_B.pkl"
        )
    },

    "Hospital C": {

        "dataset": os.path.join(
            BASE_DIR,
            "data",
            "hospital_C.csv"
        ),

        "model": os.path.join(
            MODELS_DIR,
            "hospital_C_model.pth"
        ),

        "personalized_model": os.path.join(
            MODELS_DIR,
            "personalized_hospital_C.pth"
        ),

        "scaler": os.path.join(
            MODELS_DIR,
            "scaler_C.pkl"
        )
    }
}


# ============================================================
# GLOBAL MODEL STATE
# ============================================================

model = None

MODEL_STATUS = "Not Loaded"

ACTIVE_MODEL_TYPE = "Not Loaded"

ACTIVE_MODEL_PATH = None

LAST_AGGREGATION_RESULT = None

scaler = None


# ============================================================
# GENERAL HELPERS
# ============================================================

def file_exists(path):

    return bool(
        path
        and os.path.isfile(path)
    )


def ensure_directory(path):

    if path:
        os.makedirs(
            path,
            exist_ok=True
        )


# ============================================================
# RESULT STORAGE
# ============================================================

def ensure_results_file():

    ensure_directory(
        RESULTS_DIR
    )

    if not os.path.exists(
        RESULTS_FILE
    ):

        with open(
            RESULTS_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                [],
                file,
                indent=4
            )


def load_prediction_results():

    ensure_results_file()

    try:

        with open(
            RESULTS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if isinstance(
            data,
            list
        ):

            return data

        return []

    except (
        json.JSONDecodeError,
        OSError,
        TypeError
    ) as error:

        print(
            "RESULT HISTORY LOAD ERROR:",
            str(error)
        )

        return []


def save_prediction_result(
    prediction,
    probability,
    xai_result
):

    ensure_results_file()

    role = session.get(
        "role"
    )

    username = session.get(
        "username"
    )

    hospital = session.get(
        "hospital"
    )

    if role == "hospital":

        result_hospital = hospital

    else:

        result_hospital = "Admin"

    result = {

        "id": str(
            uuid.uuid4()
        ),

        "hospital":
            result_hospital,

        "username":
            username,

        "role":
            role,

        "prediction":
            prediction,

        "probability":
            float(probability),

        "xai":
            xai_result,

        "timestamp":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    }

    with RESULTS_LOCK:

        results = (
            load_prediction_results()
        )

        results.insert(
            0,
            result
        )

        temporary_file = (
            RESULTS_FILE
            + ".tmp"
        )

        try:

            with open(
                temporary_file,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    results,
                    file,
                    indent=4
                )

            os.replace(
                temporary_file,
                RESULTS_FILE
            )

        except Exception:

            if os.path.exists(
                temporary_file
            ):

                try:
                    os.remove(
                        temporary_file
                    )
                except OSError:
                    pass

            raise

    return result


# ============================================================
# TARGET NORMALIZATION
# ============================================================

def normalize_target(y):

    numeric = pd.to_numeric(
        y,
        errors="coerce"
    )

    numeric = numeric.fillna(0)

    unique_values = set(
        numeric.astype(int).unique().tolist()
    )

    if unique_values.issubset(
        {0, 1}
    ):

        return numeric.astype(int)

    if unique_values.issubset(
        {1, 2}
    ):

        return (
            numeric.astype(int) > 1
        ).astype(int)

    return (
        numeric.astype(float) > 0
    ).astype(int)


# ============================================================
# HOSPITAL DATA LOADING
# ============================================================

def load_hospital_dataset(
    hospital_name
):

    if hospital_name not in HOSPITALS:

        raise ValueError(
            "Unknown hospital."
        )

    info = HOSPITALS[
        hospital_name
    ]

    dataset_path = info[
        "dataset"
    ]

    if not file_exists(
        dataset_path
    ):

        raise FileNotFoundError(
            f"Dataset not found for {hospital_name}: "
            f"{dataset_path}"
        )

    df = pd.read_csv(
        dataset_path
    )

    if "num" not in df.columns:

        raise ValueError(
            f"'num' target column is missing "
            f"in {hospital_name} dataset."
        )

    X = df.drop(
        "num",
        axis=1
    )

    y = normalize_target(
        df["num"]
    )

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.fillna(
        X.median(
            numeric_only=True
        )
    )

    X = X.fillna(0)

    if X.shape[1] != INPUT_FEATURE_COUNT:

        raise ValueError(
            f"{hospital_name} dataset must contain "
            f"exactly {INPUT_FEATURE_COUNT} input "
            f"features; found {X.shape[1]}."
        )

    return (
        df,
        X,
        y
    )


# ============================================================
# SAFE PYTORCH STATE DICT LOADING
# ============================================================

def load_state_dict_from_file(
    model_path
):

    if not file_exists(
        model_path
    ):

        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    state_dict = None

    try:

        state_dict = torch.load(
            model_path,
            map_location=torch.device("cpu"),
            weights_only=True
        )

    except TypeError:

        state_dict = torch.load(
            model_path,
            map_location=torch.device("cpu")
        )

    except Exception as first_error:

        print(
            "weights_only model loading failed. "
            "Trying standard PyTorch loading."
        )

        try:

            state_dict = torch.load(
                model_path,
                map_location=torch.device("cpu")
            )

        except Exception as second_error:

            raise RuntimeError(
                f"Could not load model file "
                f"{model_path}. "
                f"First error: {first_error}. "
                f"Second error: {second_error}"
            )

    if not isinstance(
        state_dict,
        dict
    ):

        raise ValueError(
            "Invalid PyTorch state dictionary."
        )

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            cleaned_key = key[
                len("module.") :
            ]

        else:

            cleaned_key = key

        cleaned_state_dict[
            cleaned_key
        ] = value

    return cleaned_state_dict


def load_mlp_from_file(
    model_path
):

    hospital_model = MLP(
        input_size=INPUT_FEATURE_COUNT
    )

    state_dict = (
        load_state_dict_from_file(
            model_path
        )
    )

    try:

        hospital_model.load_state_dict(
            state_dict,
            strict=True
        )

    except Exception as error:

        raise RuntimeError(
            f"Model architecture does not match "
            f"{model_path}: {error}"
        )

    hospital_model.eval()

    return hospital_model


# ============================================================
# HOSPITAL MODEL LOADING
# ============================================================

def load_hospital_model(
    hospital_name
):

    if hospital_name not in HOSPITALS:

        raise ValueError(
            "Unknown hospital."
        )

    info = HOSPITALS[
        hospital_name
    ]

    return load_mlp_from_file(
        info["model"]
    )


# ============================================================
# SCALER LOADING
# ============================================================

def load_hospital_scaler(
    hospital_name
):

    if hospital_name not in HOSPITALS:

        raise ValueError(
            "Unknown hospital."
        )

    info = HOSPITALS[
        hospital_name
    ]

    scaler_path = info[
        "scaler"
    ]

    if not file_exists(
        scaler_path
    ):

        raise FileNotFoundError(
            f"Scaler not found for {hospital_name}: "
            f"{scaler_path}"
        )

    return joblib.load(
        scaler_path
    )


# ============================================================
# LOGIN REQUIRED
# ============================================================

def login_required(function):

    @wraps(function)
    def decorated_function(
        *args,
        **kwargs
    ):

        if "username" not in session:

            if (
                request.path.startswith(
                    "/api/"
                )
                or request.path == "/predict"
            ):

                return jsonify({

                    "error":
                        "Login required.",

                    "redirect":
                        "/login"

                }), 401

            return redirect(
                url_for("login")
            )

        return function(
            *args,
            **kwargs
        )

    return decorated_function


# ============================================================
# ADMIN REQUIRED
# ============================================================

def admin_required(function):

    @wraps(function)
    def decorated_function(
        *args,
        **kwargs
    ):

        if "username" not in session:

            return jsonify({

                "error":
                    "Login required.",

                "redirect":
                    "/login"

            }), 401

        if session.get(
            "role"
        ) != "admin":

            return jsonify({

                "error":
                    "Admin access required."

            }), 403

        return function(
            *args,
            **kwargs
        )

    return decorated_function


# ============================================================
# XAI FEATURE NAMES
# ============================================================

FEATURE_NAMES = [

    "Age",

    "Sex",

    "Chest Pain Type",

    "Resting Blood Pressure",

    "Cholesterol",

    "Fasting Blood Sugar",

    "Resting ECG",

    "Maximum Heart Rate",

    "Exercise Induced Angina",

    "ST Depression (Oldpeak)",

    "ST Segment Slope",

    "Number of Major Vessels (CA)",

    "Thalassemia"
]


# ============================================================
# XAI - INTEGRATED GRADIENTS
# ============================================================

def explain_prediction_integrated_gradients(
    input_tensor,
    steps=50
):

    if model is None:

        raise RuntimeError(
            "Model is not loaded."
        )

    if input_tensor.ndim != 2:

        raise ValueError(
            "Integrated Gradients expects "
            "a 2-dimensional input tensor."
        )

    model.eval()

    baseline = torch.zeros_like(
        input_tensor
    )

    total_gradients = torch.zeros_like(
        input_tensor
    )

    for i in range(
        1,
        steps + 1
    ):

        alpha = (
            float(i)
            / float(steps)
        )

        interpolated = (
            baseline
            + alpha
            * (
                input_tensor
                - baseline
            )
        )

        interpolated = (
            interpolated
            .clone()
            .detach()
        )

        interpolated.requires_grad_(
            True
        )

        output = model(
            interpolated
        )

        output_scalar = (
            output
            .reshape(-1)[0]
        )

        gradient = torch.autograd.grad(
            output_scalar,
            interpolated,
            retain_graph=False,
            create_graph=False
        )[0]

        total_gradients += (
            gradient.detach()
        )

    average_gradients = (
        total_gradients
        / float(steps)
    )

    attributions = (
        input_tensor
        - baseline
    ) * average_gradients

    return (
        attributions
        .detach()
        .cpu()
        .numpy()[0]
    )


def generate_xai_result(
    values,
    X,
    probability
):

    try:

        attributions = (
            explain_prediction_integrated_gradients(
                X,
                steps=50
            )
        )

        explanations = []

        feature_count = min(
            len(FEATURE_NAMES),
            len(values),
            len(attributions)
        )

        for i in range(
            feature_count
        ):

            contribution = float(
                attributions[i]
            )

            if contribution > 0:

                impact = (
                    "Pushes toward positive "
                    "heart-disease prediction"
                )

            elif contribution < 0:

                impact = (
                    "Pushes toward negative "
                    "heart-disease prediction"
                )

            else:

                impact = (
                    "Minimal influence"
                )

            explanations.append({

                "name":
                    FEATURE_NAMES[i],

                "value":
                    float(values[i]),

                "contribution":
                    round(
                        contribution * 100,
                        4
                    ),

                "impact":
                    impact
            })

        explanations_sorted = sorted(
            explanations,
            key=lambda item:
                abs(
                    item["contribution"]
                ),
            reverse=True
        )

        top_features = (
            explanations_sorted[:5]
        )

        if probability >= 50:

            condition = (
                "The model predicts a higher "
                "likelihood of heart disease."
            )

            management = (
                "This is a model prediction and "
                "not a medical diagnosis. A qualified "
                "healthcare professional should review "
                "the patient's clinical condition."
            )

        else:

            condition = (
                "The model predicts a lower "
                "likelihood of heart disease."
            )

            management = (
                "This is a model prediction and "
                "not a medical diagnosis. Maintaining "
                "healthy lifestyle habits and routine "
                "medical check-ups can support "
                "cardiovascular health."
            )

        return {

            "method":
                "Integrated Gradients",

            "condition":
                condition,

            "management":
                management,

            "features":
                explanations,

            "top_features":
                top_features
        }

    except Exception as error:

        print(
            "XAI ERROR:",
            str(error)
        )

        return {

            "method":
                "Integrated Gradients",

            "condition":
                "Explanation unavailable.",

            "management":
                (
                    "Please consult a qualified "
                    "healthcare professional."
                ),

            "features":
                [],

            "top_features":
                []
        }


# ============================================================
# HOSPITAL EVALUATION
# ============================================================

def evaluate_hospital(
    hospital_name
):

    try:

        (
            _,
            X,
            y
        ) = load_hospital_dataset(
            hospital_name
        )

        hospital_scaler = (
            load_hospital_scaler(
                hospital_name
            )
        )

        X_scaled = (
            hospital_scaler.transform(
                X
            )
        )

        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        )

        hospital_model = (
            load_hospital_model(
                hospital_name
            )
        )

        with torch.no_grad():

            probabilities = (
                hospital_model(
                    X_tensor
                )
                .cpu()
                .numpy()
                .flatten()
            )

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        accuracy = accuracy_score(
            y,
            predictions
        )

        precision = precision_score(
            y,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y,
            predictions,
            zero_division=0
        )

        return {

            "samples":
                int(len(y)),

            "accuracy":
                round(
                    accuracy * 100,
                    2
                ),

            "precision":
                round(
                    precision * 100,
                    2
                ),

            "recall":
                round(
                    recall * 100,
                    2
                ),

            "f1":
                round(
                    f1 * 100,
                    2
                )
        }

    except Exception as error:

        print(
            f"Error evaluating "
            f"{hospital_name}: {error}"
        )

        return None


# ============================================================
# ADAPTIVE PERSONALIZED WEIGHT CALCULATION
# ============================================================

def calculate_adaptive_weights(
    evaluations
):

    valid_evaluations = {}

    for hospital in HOSPITALS:

        evaluation = evaluations.get(
            hospital
        )

        if (
            evaluation is not None
            and evaluation.get("samples", 0) > 0
        ):

            valid_evaluations[
                hospital
            ] = evaluation

    if not valid_evaluations:

        return {}

    total_samples = sum(

        max(
            0,
            int(
                evaluation.get(
                    "samples",
                    0
                )
            )
        )

        for evaluation
        in valid_evaluations.values()
    )

    if total_samples <= 0:

        equal_weight = (
            100.0
            / len(valid_evaluations)
        )

        return normalize_weights({

            hospital:
                equal_weight

            for hospital
            in valid_evaluations
        })

    raw_scores = {}

    for hospital, evaluation in (
        valid_evaluations.items()
    ):

        accuracy_value = max(
            0.0,
            min(
                1.0,
                float(
                    evaluation.get(
                        "accuracy",
                        0
                    )
                ) / 100.0
            )
        )

        f1_value = max(
            0.0,
            min(
                1.0,
                float(
                    evaluation.get(
                        "f1",
                        0
                    )
                ) / 100.0
            )
        )

        quality_score = (
            0.60
            * accuracy_value
            +
            0.40
            * f1_value
        )

        participation_score = (
            float(
                evaluation.get(
                    "samples",
                    0
                )
            )
            / float(total_samples)
        )

        adaptive_score = (
            0.70
            * quality_score
            +
            0.30
            * participation_score
        )

        raw_scores[
            hospital
        ] = max(
            0.0,
            adaptive_score
        )

    total_score = sum(
        raw_scores.values()
    )

    if total_score <= 0:

        return normalize_weights({

            hospital:
                1.0

            for hospital
            in raw_scores
        })

    weights = {

        hospital:
            (
                score
                / total_score
            ) * 100.0

        for hospital, score
        in raw_scores.items()
    }

    return normalize_weights(
        weights
    )


# ============================================================
# SAVE ADAPTIVE WEIGHTS
# ============================================================

def save_adaptive_weights(
    weights
):

    ensure_directory(
        MODELS_DIR
    )

    weights = normalize_weights(
        weights
    )

    with open(
        WEIGHTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "Adaptive Personalized Aggregation Weights\n"
        )

        file.write(
            "Formula: 70% model quality + "
            "30% data participation\n"
        )

        file.write(
            "Model quality: 60% accuracy + "
            "40% F1-score\n\n"
        )

        for hospital in HOSPITALS:

            weight = weights.get(
                hospital,
                0.0
            )

            file.write(
                f"{hospital}: {weight:.2f}\n"
            )


# ============================================================
# READ ADAPTIVE WEIGHTS
# ============================================================

def read_weight_file():

    if not file_exists(
        WEIGHTS_PATH
    ):

        return {}

    weights = {}

    try:

        with open(
            WEIGHTS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                match = re.match(
                    r"^(Hospital\s+[ABC])\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*$",
                    line,
                    re.IGNORECASE
                )

                if match:

                    hospital_letter = (
                        match.group(1)
                        .strip()
                        .upper()
                    )

                    hospital = (
                        hospital_letter
                        .replace(
                            "HOSPITAL ",
                            "Hospital "
                        )
                    )

                    value = float(
                        match.group(2)
                    )

                    if value >= 0:

                        weights[
                            hospital
                        ] = value

    except Exception as error:

        print(
            "WEIGHT FILE ERROR:",
            str(error)
        )

        return {}

    return weights


# ============================================================
# NORMALIZE WEIGHTS
# ============================================================

def normalize_weights(
    weights
):

    cleaned = {}

    if not isinstance(
        weights,
        dict
    ):

        return {}

    for hospital, value in (
        weights.items()
    ):

        if hospital not in HOSPITALS:

            continue

        try:

            numeric_value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            numeric_value = 0.0

        if (
            numeric_value < 0
            or pd.isna(
                numeric_value
            )
        ):

            numeric_value = 0.0

        cleaned[
            hospital
        ] = numeric_value

    if not cleaned:

        return {}

    total = sum(
        cleaned.values()
    )

    if total <= 0:

        equal_weight = (
            100.0
            / len(cleaned)
        )

        normalized = {

            hospital:
                equal_weight

            for hospital
            in cleaned
        }

    else:

        normalized = {

            hospital:
                (
                    value
                    / total
                ) * 100.0

            for hospital, value
            in cleaned.items()
        }

    rounded = {

        hospital:
            round(
                value,
                2
            )

        for hospital, value
        in normalized.items()
    }

    difference = round(
        100.0
        - sum(
            rounded.values()
        ),
        2
    )

    if rounded:

        last_hospital = list(
            rounded.keys()
        )[-1]

        rounded[
            last_hospital
        ] = round(
            rounded[
                last_hospital
            ] + difference,
            2
        )

    return rounded


# ============================================================
# GET ADAPTIVE WEIGHTS
# ============================================================

def get_adaptive_weights():

    weights = read_weight_file()

    if not weights:

        return {
            "Hospital A": 33.33,
            "Hospital B": 33.33,
            "Hospital C": 33.34
        }

    complete_weights = {

        hospital:
            weights.get(
                hospital,
                0.0
            )

        for hospital
        in HOSPITALS
    }

    return normalize_weights(
        complete_weights
    )


# ============================================================
# STATE DICT COMPATIBILITY CHECK
# ============================================================

def validate_state_dict_compatibility(
    state_dicts
):

    if not state_dicts:

        return False, (
            "No state dictionaries available."
        )

    hospitals = list(
        state_dicts.keys()
    )

    reference_hospital = (
        hospitals[0]
    )

    reference_state = (
        state_dicts[
            reference_hospital
        ]
    )

    reference_keys = set(
        reference_state.keys()
    )

    for hospital in hospitals[1:]:

        current_state = (
            state_dicts[
                hospital
            ]
        )

        current_keys = set(
            current_state.keys()
        )

        if current_keys != reference_keys:

            missing = (
                reference_keys
                - current_keys
            )

            extra = (
                current_keys
                - reference_keys
            )

            return False, (

                f"Model structure mismatch "
                f"between {reference_hospital} "
                f"and {hospital}. "
                f"Missing keys: {missing}. "
                f"Extra keys: {extra}."
            )

        for key in reference_state:

            if (
                not torch.is_tensor(
                    reference_state[key]
                )
                or not torch.is_tensor(
                    current_state[key]
                )
            ):

                return False, (
                    f"Invalid tensor for parameter "
                    f"{key}."
                )

            if (
                reference_state[key].shape
                !=
                current_state[key].shape
            ):

                return False, (

                    f"Shape mismatch for "
                    f"{key}: "
                    f"{reference_state[key].shape} "
                    f"vs "
                    f"{current_state[key].shape}"
                )

    return True, "Compatible"


# ============================================================
# STATE DICT AGGREGATION
# ============================================================

def aggregate_state_dicts(
    state_dicts,
    weights
):

    if not state_dicts:

        raise ValueError(
            "No hospital model state dictionaries available."
        )

    participating = [

        hospital

        for hospital
        in state_dicts

        if hospital in weights
        and float(
            weights[hospital]
        ) > 0
    ]

    if not participating:

        raise ValueError(
            "No participating hospital models found."
        )

    compatibility_ok, message = (
        validate_state_dict_compatibility({
            hospital:
                state_dicts[hospital]

            for hospital
            in participating
        })
    )

    if not compatibility_ok:

        raise ValueError(
            message
        )

    total_weight = sum(

        float(
            weights[hospital]
        )

        for hospital
        in participating
    )

    if total_weight <= 0:

        raise ValueError(
            "Adaptive aggregation weights are invalid."
        )

    normalized_decimal_weights = {

        hospital:
            (
                float(
                    weights[hospital]
                )
                / total_weight
            )

        for hospital
        in participating
    }

    reference_state = (
        state_dicts[
            participating[0]
        ]
    )

    aggregated_state = {}

    for key in reference_state:

        reference_tensor = (
            reference_state[key]
        )

        if torch.is_floating_point(
            reference_tensor
        ):

            aggregate_tensor = (
                torch.zeros_like(
                    reference_tensor
                )
            )

            for hospital in participating:

                hospital_tensor = (
                    state_dicts[
                        hospital
                    ][key]
                )

                aggregate_tensor += (
                    hospital_tensor
                    *
                    normalized_decimal_weights[
                        hospital
                    ]
                )

            aggregated_state[
                key
            ] = aggregate_tensor

        else:

            # Integer/buffer parameters cannot be
            # averaged safely. Use the reference value.
            aggregated_state[
                key
            ] = reference_tensor.clone()

    return aggregated_state


# ============================================================
# ACTUAL ADAPTIVE PERSONALIZED AGGREGATION
# ============================================================

def perform_adaptive_aggregation():

    global LAST_AGGREGATION_RESULT

    print("=" * 70)

    print(
        "STARTING ADAPTIVE PERSONALIZED AGGREGATION"
    )

    print("=" * 70)

    evaluations = {}

    state_dicts = {}

    for hospital in HOSPITALS:

        evaluation = evaluate_hospital(
            hospital
        )

        if evaluation is None:

            print(
                f"Skipping {hospital}: "
                "evaluation unavailable."
            )

            continue

        model_path = HOSPITALS[
            hospital
        ]["model"]

        if not file_exists(
            model_path
        ):

            print(
                f"Skipping {hospital}: "
                "model file unavailable."
            )

            continue

        try:

            state_dict = (
                load_state_dict_from_file(
                    model_path
                )
            )

            evaluations[
                hospital
            ] = evaluation

            state_dicts[
                hospital
            ] = state_dict

        except Exception as error:

            print(
                f"Could not load {hospital} "
                f"model: {error}"
            )

    if not state_dicts:

        result = {

            "success":
                False,

            "message":
                "No hospital models were available.",

            "weights":
                {},

            "evaluations":
                {},

            "participating_hospitals":
                [],

            "model_path":
                None,

            "aggregation_method":
                AGGREGATION_METHOD,

            "algorithm":
                AGGREGATION_ALGORITHM
        }

        LAST_AGGREGATION_RESULT = result

        return result

    weights = calculate_adaptive_weights(
        evaluations
    )

    if not weights:

        result = {

            "success":
                False,

            "message":
                "Adaptive weights could not be calculated.",

            "weights":
                {},

            "evaluations":
                evaluations,

            "participating_hospitals":
                list(
                    state_dicts.keys()
                ),

            "model_path":
                None,

            "aggregation_method":
                AGGREGATION_METHOD,

            "algorithm":
                AGGREGATION_ALGORITHM
        }

        LAST_AGGREGATION_RESULT = result

        return result

    participating_weights = {

        hospital:
            weights.get(
                hospital,
                0.0
            )

        for hospital
        in state_dicts
    }

    participating_weights = normalize_weights(
        participating_weights
    )

    try:

        aggregated_state = (
            aggregate_state_dicts(
                state_dicts,
                participating_weights
            )
        )

    except Exception as error:

        result = {

            "success":
                False,

            "message":
                (
                    "Adaptive aggregation failed "
                    f"during model aggregation: {error}"
                ),

            "weights":
                participating_weights,

            "evaluations":
                evaluations,

            "participating_hospitals":
                list(
                    state_dicts.keys()
                ),

            "model_path":
                None,

            "aggregation_method":
                AGGREGATION_METHOD,

            "algorithm":
                AGGREGATION_ALGORITHM
        }

        LAST_AGGREGATION_RESULT = result

        return result

    ensure_directory(
        MODELS_DIR
    )

    temporary_model_path = (
        ADAPTIVE_GLOBAL_MODEL_PATH
        + ".tmp"
    )

    try:

        torch.save(
            aggregated_state,
            temporary_model_path
        )

        os.replace(
            temporary_model_path,
            ADAPTIVE_GLOBAL_MODEL_PATH
        )

    except Exception as error:

        if os.path.exists(
            temporary_model_path
        ):

            try:
                os.remove(
                    temporary_model_path
                )
            except OSError:
                pass

        result = {

            "success":
                False,

            "message":
                (
                    "Adaptive global model could "
                    f"not be saved: {error}"
                ),

            "weights":
                participating_weights,

            "evaluations":
                evaluations,

            "participating_hospitals":
                list(
                    state_dicts.keys()
                ),

            "model_path":
                None,

            "aggregation_method":
                AGGREGATION_METHOD,

            "algorithm":
                AGGREGATION_ALGORITHM
        }

        LAST_AGGREGATION_RESULT = result

        return result

    save_adaptive_weights(
        participating_weights
    )

    result = {

        "success":
            True,

        "message":
            (
                "Adaptive Personalized Aggregation "
                "completed successfully."
            ),

        "weights":
            participating_weights,

        "evaluations":
            evaluations,

        "participating_hospitals":
            list(
                state_dicts.keys()
            ),

        "model_path":
            ADAPTIVE_GLOBAL_MODEL_PATH,

        "aggregation_method":
            AGGREGATION_METHOD,

        "algorithm":
            AGGREGATION_ALGORITHM
    }

    LAST_AGGREGATION_RESULT = result

    print(
        "Adaptive weights:"
    )

    for hospital in HOSPITALS:

        print(
            f"  {hospital}: "
            f"{participating_weights.get(hospital, 0.0):.2f}%"
        )

    print(
        "Adaptive global model saved:"
    )

    print(
        ADAPTIVE_GLOBAL_MODEL_PATH
    )

    print("=" * 70)

    print(
        "ADAPTIVE AGGREGATION COMPLETED"
    )

    print("=" * 70)

    return result
# ============================================================
# ACTIVE GLOBAL MODEL SELECTION
# ============================================================

def select_active_model_path():

    # Proposed adaptive model gets the highest priority.
    # This is the final model produced by the proposed
    # adaptive quality-based federated training.

    if file_exists(
        PROPOSED_ADAPTIVE_MODEL_PATH
    ):

        return (
            PROPOSED_ADAPTIVE_MODEL_PATH,
            "Proposed Adaptive Quality-Based Global Model"
        )

    # Keep the previous adaptive model as a fallback.

    if file_exists(
        ADAPTIVE_GLOBAL_MODEL_PATH
    ):

        return (
            ADAPTIVE_GLOBAL_MODEL_PATH,
            "Adaptive Personalized Global Model"
        )

    # Private global model fallback.

    if file_exists(
        PRIVATE_GLOBAL_MODEL_PATH
    ):

        return (
            PRIVATE_GLOBAL_MODEL_PATH,
            "Private Global Model"
        )

    # Legacy model fallback.

    if file_exists(
        LEGACY_GLOBAL_MODEL_PATH
    ):

        return (
            LEGACY_GLOBAL_MODEL_PATH,
            "Legacy Global Model"
        )

    return (
        None,
        "Not Loaded"
    )


def initialize_global_model():

    global model
    global MODEL_STATUS
    global ACTIVE_MODEL_TYPE
    global ACTIVE_MODEL_PATH

    model_path, model_type = (
        select_active_model_path()
    )

    if model_path is None:

        model = None

        MODEL_STATUS = (
            "Not Loaded"
        )

        ACTIVE_MODEL_TYPE = (
            "Not Loaded"
        )

        ACTIVE_MODEL_PATH = None

        print(
            "No global model was found."
        )

        return False

    try:

        loaded_model = (
            load_mlp_from_file(
                model_path
            )
        )

        model = loaded_model

        ACTIVE_MODEL_PATH = (
            model_path
        )

        ACTIVE_MODEL_TYPE = (
            model_type
        )

        MODEL_STATUS = (
            "Loaded"
        )

        print("=" * 70)

        print(
            "GLOBAL FEDHEALTH-XAI MODEL LOADED"
        )

        print(
            "Active model:",
            ACTIVE_MODEL_TYPE
        )

        print(
            "Model path:",
            ACTIVE_MODEL_PATH
        )

        print("=" * 70)

        return True

    except Exception as error:

        model = None

        MODEL_STATUS = (
            "Error"
        )

        ACTIVE_MODEL_TYPE = (
            "Not Loaded"
        )

        ACTIVE_MODEL_PATH = None

        print(
            "MODEL LOADING ERROR:",
            str(error)
        )

        return False
# ============================================================
# HOSPITAL STATUS
# ============================================================

def get_hospital_status():

    status = {}

    for hospital, info in (
        HOSPITALS.items()
    ):

        dataset_exists = (
            file_exists(
                info["dataset"]
            )
        )

        model_exists = (
            file_exists(
                info["model"]
            )
        )

        personalized_exists = (
            file_exists(
                info["personalized_model"]
            )
        )

        scaler_exists = (
            file_exists(
                info["scaler"]
            )
        )

        status[hospital] = {

            "dataset":
                dataset_exists,

            "model":
                model_exists,

            "personalized_model":
                personalized_exists,

            "scaler":
                scaler_exists,

            "status":
                (
                    "Ready"
                    if (
                        dataset_exists
                        and model_exists
                        and scaler_exists
                    )
                    else
                    "Not Ready"
                )
        }

    return status


# ============================================================
# FEDERATED STATUS
# ============================================================

def get_federated_status():

    active_model_exists = (
        file_exists(
            ACTIVE_MODEL_PATH
        )
        if ACTIVE_MODEL_PATH
        else False
    )

    return {

        "global_model":
            active_model_exists,

        "model_status":
            MODEL_STATUS,

        "active_model_type":
            ACTIVE_MODEL_TYPE,

        "hospitals":
            get_hospital_status(),

        "weights":
            get_adaptive_weights(),

        "aggregation_method":
            AGGREGATION_METHOD,

        "algorithm":
            AGGREGATION_ALGORITHM,

        "training_rounds":
            TRAINING_ROUNDS
    }


# ============================================================
# MODEL ANALYTICS
# ============================================================

def get_model_analytics():

    hospitals_data = {}

    for hospital_name in HOSPITALS:

        result = evaluate_hospital(
            hospital_name
        )

        if result is not None:

            hospitals_data[
                hospital_name
            ] = result

    global_true = []

    global_predictions = []

    if model is not None:

        model.eval()

        for hospital_name in HOSPITALS:

            try:

                (
                    _,
                    X,
                    y
                ) = load_hospital_dataset(
                    hospital_name
                )

                hospital_scaler = (
                    load_hospital_scaler(
                        hospital_name
                    )
                )

                X_scaled = (
                    hospital_scaler.transform(
                        X
                    )
                )

                X_tensor = torch.tensor(
                    X_scaled,
                    dtype=torch.float32
                )

                with torch.no_grad():

                    probabilities = (
                        model(
                            X_tensor
                        )
                        .cpu()
                        .numpy()
                        .flatten()
                    )

                predictions = (
                    probabilities >= 0.5
                ).astype(int)

                global_true.extend(
                    y.astype(int).tolist()
                )

                global_predictions.extend(
                    predictions.tolist()
                )

            except Exception as error:

                print(
                    "Global model evaluation "
                    f"error for {hospital_name}: "
                    f"{error}"
                )

    if global_true:

        global_accuracy = round(
            accuracy_score(
                global_true,
                global_predictions
            ) * 100,
            2
        )

        global_precision = round(
            precision_score(
                global_true,
                global_predictions,
                zero_division=0
            ) * 100,
            2
        )

        global_recall = round(
            recall_score(
                global_true,
                global_predictions,
                zero_division=0
            ) * 100,
            2
        )

        global_f1 = round(
            f1_score(
                global_true,
                global_predictions,
                zero_division=0
            ) * 100,
            2
        )

    else:

        global_accuracy = None

        global_precision = None

        global_recall = None

        global_f1 = None

    return {

        "global_accuracy":
            global_accuracy,

        "global_precision":
            global_precision,

        "global_recall":
            global_recall,

        "global_f1":
            global_f1,

        "hospitals_data":
            hospitals_data
    }


# ============================================================
# ROLE-BASED ANALYTICS
# ============================================================

def get_visible_analytics():

    analytics = (
        get_model_analytics()
    )

    role = session.get(
        "role"
    )

    hospital = session.get(
        "hospital"
    )

    if role == "admin":

        return analytics

    hospital_data = {}

    if hospital in analytics[
        "hospitals_data"
    ]:

        hospital_data[
            hospital
        ] = analytics[
            "hospitals_data"
        ][hospital]

    return {

        "global_accuracy":
            analytics[
                "global_accuracy"
            ],

        "global_precision":
            analytics[
                "global_precision"
            ],

        "global_recall":
            analytics[
                "global_recall"
            ],

        "global_f1":
            analytics[
                "global_f1"
            ],

        "hospitals_data":
            hospital_data
    }


# ============================================================
# DASHBOARD DATA
# ============================================================

def get_dashboard_data():

    weights = (
        get_adaptive_weights()
    )

    hospital_status = (
        get_hospital_status()
    )

    analytics = (
        get_visible_analytics()
    )

    active_model_exists = (
        file_exists(
            ACTIVE_MODEL_PATH
        )
        if ACTIVE_MODEL_PATH
        else False
    )

    return {

        "model_status":
            MODEL_STATUS,

        "global_model":
            active_model_exists,

        "active_model_type":
            ACTIVE_MODEL_TYPE,

        "hospitals":
            len(HOSPITALS),

        "rounds":
            TRAINING_ROUNDS,

        "accuracy":
            analytics[
                "global_accuracy"
            ],

        "global_precision":
            analytics[
                "global_precision"
            ],

        "global_recall":
            analytics[
                "global_recall"
            ],

        "global_f1":
            analytics[
                "global_f1"
            ],

        "hospitals_data":
            analytics[
                "hospitals_data"
            ],

        "weights":
            weights,

        "hospital_status":
            hospital_status,

        "aggregation_method":
            AGGREGATION_METHOD,

        "algorithm":
            AGGREGATION_ALGORITHM,

        "timestamp":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    }


# ============================================================
# PREDICTION HELPER
# ============================================================

def make_prediction(
    values,
    prediction_scaler=None
):

    if model is None:

        raise RuntimeError(
            "FedHealth-XAI global model is not loaded."
        )

    if len(values) != INPUT_FEATURE_COUNT:

        raise ValueError(
            f"Exactly {INPUT_FEATURE_COUNT} "
            "patient values are required."
        )

    active_scaler = (
        prediction_scaler
        if prediction_scaler is not None
        else scaler
    )

    if active_scaler is None:

        raise RuntimeError(
            "Prediction scaler is not loaded."
        )

    try:

        scaled_values = (
            active_scaler.transform(
                [values]
            )
        )

    except Exception as error:

        raise ValueError(
            f"Could not scale patient values: "
            f"{error}"
        )

    X = torch.tensor(
        scaled_values,
        dtype=torch.float32
    )

    model.eval()

    with torch.no_grad():

        output = model(
            X
        )

    probability = float(
        output
        .detach()
        .cpu()
        .reshape(-1)[0]
        .item()
    )

    probability = max(
        0.0,
        min(
            1.0,
            probability
        )
    )

    probability_percent = round(
        probability * 100,
        2
    )

    prediction = (
        "Positive"
        if probability >= 0.5
        else "Negative"
    )

    return (
        prediction,
        probability_percent,
        X
    )


# ============================================================
# GLOBAL SCALER LOADING
# ============================================================

def initialize_prediction_scaler():

    global scaler

    scaler = None

    try:

        if file_exists(
            SCALER_PATH
        ):

            scaler = joblib.load(
                SCALER_PATH
            )

            print(
                "Prediction scaler loaded successfully."
            )

            return True

        print(
            "Prediction scaler file not found:"
        )

        print(
            SCALER_PATH
        )

        return False

    except Exception as error:

        scaler = None

        print(
            "SCALER LOADING ERROR:",
            str(error)
        )

        return False


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if "username" in session:

        return redirect(
            url_for("home")
        )

    error = None

    if request.method == "POST":

        username = (
            request.form.get(
                "username",
                ""
            )
            .strip()
            .lower()
        )

        password = request.form.get(
            "password",
            ""
        )

        user = USERS.get(
            username
        )

        if (
            user is not None
            and user["password"] == password
        ):

            session.clear()

            session["username"] = (
                username
            )

            session["role"] = (
                user["role"]
            )

            session["hospital"] = (
                user["hospital"]
            )

            print(
                "LOGIN SUCCESS:",
                username,
                "| Role:",
                user["role"],
                "| Hospital:",
                user["hospital"]
            )

            return redirect(
                url_for("home")
            )

        error = (
            "Invalid username or password."
        )

    return render_template(
        "login.html",
        error=error
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route(
    "/logout",
    methods=["GET"]
)
def logout():

    username = session.get(
        "username"
    )

    session.clear()

    print(
        "Logged out:",
        username
    )

    return redirect(
        url_for("login")
    )


# ============================================================
# HOME / DASHBOARD
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
@login_required
def home():

    data = (
        get_dashboard_data()
    )

    weights = (
        get_adaptive_weights()
    )

    return render_template(

        "dashboard.html",

        weight_A=
            weights.get(
                "Hospital A",
                33.33
            ),

        weight_B=
            weights.get(
                "Hospital B",
                33.33
            ),

        weight_C=
            weights.get(
                "Hospital C",
                33.34
            ),

        weights=
            weights,

        prediction=None,

        probability=None,

        error=None,

        dashboard_data=
            data,

        current_user=
            session.get(
                "username"
            ),

        user_role=
            session.get(
                "role"
            ),

        user_hospital=
            session.get(
                "hospital"
            ),

        aggregation_method=
            AGGREGATION_METHOD,

        aggregation_algorithm=
            AGGREGATION_ALGORITHM
    )


# ============================================================
# PATIENT PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
@login_required
def predict():

    try:

        if model is None:

            return jsonify({

                "error":
                    "FedHealth-XAI global model "
                    "is not loaded."

            }), 500

        fields = [

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

        values = []

        for field in fields:

            raw_value = request.form.get(
                field
            )

            if (
                raw_value is None
                or str(
                    raw_value
                ).strip() == ""
            ):

                return jsonify({

                    "error":
                        f"Please enter a value "
                        f"for {field}."

                }), 400

            try:

                numeric_value = float(
                    raw_value
                )

            except (
                TypeError,
                ValueError
            ):

                return jsonify({

                    "error":
                        f"Invalid value "
                        f"for {field}."

                }), 400

            if pd.isna(
                numeric_value
            ):

                return jsonify({

                    "error":
                        f"Invalid value "
                        f"for {field}."

                }), 400

            values.append(
                numeric_value
            )

        # ----------------------------------------------------
        # Select the appropriate scaler.
        # Hospital users use their own hospital scaler.
        # Admin uses the existing global/prediction scaler.
        # ----------------------------------------------------

        prediction_scaler = scaler

        if session.get(
            "role"
        ) == "hospital":

            hospital_name = (
                session.get(
                    "hospital"
                )
            )

            hospital_info = (
                HOSPITALS.get(
                    hospital_name
                )
            )

            if hospital_info is None:

                return jsonify({

                    "error":
                        "Hospital information "
                        "is unavailable."

                }), 500

            hospital_scaler_path = (
                hospital_info[
                    "scaler"
                ]
            )

            if not file_exists(
                hospital_scaler_path
            ):

                return jsonify({

                    "error":
                        "Hospital prediction "
                        "scaler is not available."

                }), 500

            try:

                prediction_scaler = (
                    joblib.load(
                        hospital_scaler_path
                    )
                )

            except Exception as error:

                return jsonify({

                    "error":
                        "Hospital scaler could "
                        f"not be loaded: {error}"

                }), 500

        if prediction_scaler is None:

            return jsonify({

                "error":
                    "Prediction scaler "
                    "is not loaded."

            }), 500

        print("=" * 70)

        print(
            "PATIENT PREDICTION REQUEST"
        )

        print(
            "Logged-in user:",
            session.get(
                "username"
            )
        )

        print(
            "Role:",
            session.get(
                "role"
            )
        )

        print(
            "Hospital:",
            session.get(
                "hospital"
            )
        )

        print(
            "Active model:",
            ACTIVE_MODEL_TYPE
        )

        print(
            "Patient values:",
            values
        )

        prediction, probability_percent, X = (
            make_prediction(
                values,
                prediction_scaler
            )
        )

        print(
            "Prediction probability:",
            probability_percent,
            "%"
        )

        xai_result = (
            generate_xai_result(
                values,
                X,
                probability_percent
            )
        )

        saved_result = (
            save_prediction_result(
                prediction,
                probability_percent,
                xai_result
            )
        )

        print(
            "Prediction:",
            prediction
        )

        print(
            "Result saved for:",
            saved_result[
                "hospital"
            ]
        )

        print(
            "XAI Method:",
            xai_result[
                "method"
            ]
        )

        print("=" * 70)

        return jsonify({

            "prediction":
                prediction,

            "probability":
                probability_percent,

            "xai":
                xai_result,

            "saved":
                True,

            "hospital":
                saved_result[
                    "hospital"
                ],

            "username":
                saved_result[
                    "username"
                ],

            "timestamp":
                saved_result[
                    "timestamp"
                ],

            "model":
                ACTIVE_MODEL_TYPE,

            "aggregation_method":
                AGGREGATION_METHOD,

            "aggregation_algorithm":
                AGGREGATION_ALGORITHM
        })

    except Exception as error:

        print("=" * 70)

        print(
            "PREDICTION ERROR"
        )

        print(
            str(error)
        )

        print("=" * 70)

        return jsonify({

            "error":
                str(error)

        }), 400


# ============================================================
# API - PREDICTION RESULT HISTORY
# ============================================================

@app.route(
    "/api/results",
    methods=["GET"]
)
@login_required
def api_results():

    results = (
        load_prediction_results()
    )

    role = session.get(
        "role"
    )

    hospital = session.get(
        "hospital"
    )

    if role == "admin":

        visible_results = (
            results
        )

        scope = (
            "All Hospitals"
        )

    else:

        visible_results = [

            result

            for result in results

            if result.get(
                "hospital"
            ) == hospital
        ]

        scope = hospital

    return jsonify({

        "results":
            visible_results,

        "count":
            len(
                visible_results
            ),

        "scope":
            scope,

        "role":
            role,

        "hospital":
            hospital
    })


# ============================================================
# API - GENERAL STATUS
# ============================================================

@app.route(
    "/api/status",
    methods=["GET"]
)
@login_required
def api_status():

    return jsonify(
        get_federated_status()
    )


# ============================================================
# API - HOSPITAL STATUS
# ============================================================

@app.route(
    "/api/hospital/<hospital_name>",
    methods=["GET"]
)
@login_required
def api_hospital(
    hospital_name
):

    hospital_name = (
        hospital_name.replace(
            "_",
            " "
        )
    )

    matched_hospital = None

    for hospital in HOSPITALS:

        if (
            hospital.lower()
            == hospital_name.lower()
        ):

            matched_hospital = (
                hospital
            )

            break

    if matched_hospital is None:

        return jsonify({

            "error":
                "Hospital not found."

        }), 404

    if (
        session.get(
            "role"
        ) == "hospital"
    ):

        if (
            session.get(
                "hospital"
            )
            != matched_hospital
        ):

            return jsonify({

                "error":
                    "Access denied. "
                    "You can only access "
                    "your hospital."

            }), 403

    info = HOSPITALS[
        matched_hospital
    ]

    return jsonify({

        "hospital":
            matched_hospital,

        "dataset":
            file_exists(
                info["dataset"]
            ),

        "model":
            file_exists(
                info["model"]
            ),

        "personalized_model":
            file_exists(
                info["personalized_model"]
            ),

        "scaler":
            file_exists(
                info["scaler"]
            ),

        "status":
            (
                "Ready"
                if (
                    file_exists(
                        info["dataset"]
                    )
                    and file_exists(
                        info["model"]
                    )
                    and file_exists(
                        info["scaler"]
                    )
                )
                else
                "Not Ready"
            )
    })


# ============================================================
# API - ADAPTIVE AGGREGATION
# ============================================================

@app.route(
    "/api/aggregation",
    methods=["GET"]
)
@login_required
def api_aggregation():

    weights = (
        get_adaptive_weights()
    )

    return jsonify({

        "method":
            AGGREGATION_METHOD,

        "algorithm":
            AGGREGATION_ALGORITHM,

        "weights":
            weights,

        "global_model":
            (
                file_exists(
                    ACTIVE_MODEL_PATH
                )
                if ACTIVE_MODEL_PATH
                else False
            ),

        "active_model":
            ACTIVE_MODEL_TYPE,

        "status":
            MODEL_STATUS,

        "training_rounds":
            TRAINING_ROUNDS
    })


# ============================================================
# API - RUN ADAPTIVE AGGREGATION
# ============================================================

@app.route(
    "/api/aggregation/run",
    methods=["POST"]
)
@admin_required
def api_run_aggregation():

    try:

        result = (
            perform_adaptive_aggregation()
        )

        if not result.get(
            "success"
        ):

            return jsonify(
                result
            ), 500

        if not initialize_global_model():

            return jsonify({

                "success":
                    False,

                "error":
                    (
                        "Aggregation completed, "
                        "but the new adaptive global "
                        "model could not be loaded."
                    ),

                "aggregation":
                    result

            }), 500

        result[
            "active_model"
        ] = ACTIVE_MODEL_TYPE

        result[
            "model_status"
        ] = MODEL_STATUS

        return jsonify(
            result
        )

    except Exception as error:

        print(
            "AGGREGATION API ERROR:",
            str(error)
        )

        return jsonify({

            "success":
                False,

            "error":
                str(error)

        }), 500


# ============================================================
# API - PRIVACY
# ============================================================

@app.route(
    "/api/privacy",
    methods=["GET"]
)
@login_required
def api_privacy():

    privacy_files = [

        os.path.join(
            BASE_DIR,
            "src",
            "privacy.py"
        ),

        os.path.join(
            BASE_DIR,
            "src",
            "dp_utils.py"
        )
    ]

    privacy_available = any(

        os.path.exists(
            path
        )

        for path in privacy_files
    )

    return jsonify({

        "differential_privacy":
            privacy_available,

        "method":
            "Clip + Add Noise",

        "status":
            (
                "Enabled"
                if privacy_available
                else
                "Not Detected"
            )
    })


# ============================================================
# API - FEDERATED LEARNING
# ============================================================

@app.route(
    "/api/federated",
    methods=["GET"]
)
@login_required
def api_federated():

    hospitals = (
        get_hospital_status()
    )

    ready_count = sum(

        1

        for info
        in hospitals.values()

        if info["status"]
        == "Ready"
    )

    return jsonify({

        "algorithm":
            AGGREGATION_ALGORITHM,

        "aggregation_method":
            AGGREGATION_METHOD,

        "hospitals":
            len(HOSPITALS),

        "ready_hospitals":
            ready_count,

        "global_model":
            (
                file_exists(
                    ACTIVE_MODEL_PATH
                )
                if ACTIVE_MODEL_PATH
                else False
            ),

        "active_model":
            ACTIVE_MODEL_TYPE,

        "status":
            (
                "Ready"
                if ready_count
                == len(HOSPITALS)
                else
                "Partial"
            )
    })


# ============================================================
# API - MODEL
# ============================================================

@app.route(
    "/api/model",
    methods=["GET"]
)
@login_required
def api_model():

    return jsonify({

        "model":
            "MLP",

        "input_features":
            INPUT_FEATURE_COUNT,

        "framework":
            "PyTorch",

        "global_model":
            (
                file_exists(
                    ACTIVE_MODEL_PATH
                )
                if ACTIVE_MODEL_PATH
                else False
            ),

        "status":
            MODEL_STATUS,

        "active_model_type":
            ACTIVE_MODEL_TYPE,

        "aggregation_method":
            AGGREGATION_METHOD,

        "aggregation_algorithm":
            AGGREGATION_ALGORITHM,

        "model_path":
            (
                os.path.relpath(
                    ACTIVE_MODEL_PATH,
                    BASE_DIR
                )
                if ACTIVE_MODEL_PATH
                else None
            )
    })


# ============================================================
# API - MODEL METRICS
# ============================================================

@app.route(
    "/api/model-metrics",
    methods=["GET"]
)
@login_required
def api_model_metrics():

    analytics = (
        get_visible_analytics()
    )

    return jsonify({

        "global_accuracy":
            analytics[
                "global_accuracy"
            ],

        "global_precision":
            analytics[
                "global_precision"
            ],

        "global_recall":
            analytics[
                "global_recall"
            ],

        "global_f1":
            analytics[
                "global_f1"
            ],

        "hospitals_data":
            analytics[
                "hospitals_data"
            ],

        "aggregation_method":
            AGGREGATION_METHOD,

        "aggregation_algorithm":
            AGGREGATION_ALGORITHM
    })


# ============================================================
# API - DASHBOARD DATA
# ============================================================

@app.route(
    "/api/dashboard",
    methods=["GET"]
)
@login_required
def api_dashboard():

    return jsonify(
        get_dashboard_data()
    )


# ============================================================
# API - LAST AGGREGATION
# ============================================================

@app.route(
    "/api/aggregation/latest",
    methods=["GET"]
)
@login_required
def api_latest_aggregation():

    if LAST_AGGREGATION_RESULT is None:

        return jsonify({

            "available":
                False,

            "message":
                "No aggregation has been run "
                "during this application session.",

            "weights":
                get_adaptive_weights(),

            "aggregation_method":
                AGGREGATION_METHOD,

            "algorithm":
                AGGREGATION_ALGORITHM
        })

    return jsonify({

        "available":
            True,

        **LAST_AGGREGATION_RESULT
    })


# ============================================================
# OPEN BROWSER
# ============================================================

def open_browser():

    try:

        webbrowser.open(
            "http://127.0.0.1:5000"
        )

    except Exception as error:

        print(
            "BROWSER OPEN ERROR:",
            str(error)
        )


# ============================================================
# STARTUP INITIALIZATION
# ============================================================

def initialize_application():

    ensure_directory(
        RESULTS_DIR
    )

    ensure_directory(
        MODELS_DIR
    )

    ensure_results_file()

    initialize_prediction_scaler()

    print("=" * 70)

    print(
        "FedHealth-XAI INITIALIZATION"
    )

    print("=" * 70)

    print(
        "Training rounds:",
        TRAINING_ROUNDS
    )

    print(
        "Participating hospitals:",
        len(HOSPITALS)
    )

    print(
        "Aggregation method:",
        AGGREGATION_METHOD
    )

    print(
        "Algorithm:",
        AGGREGATION_ALGORITHM
    )

    print(
        "Weight formula:",
        "70% model quality + 30% data participation"
    )

    print(
        "Quality formula:",
        "60% Accuracy + 40% F1-score"
    )

    print("-" * 70)

    all_hospital_models_available = all(

        file_exists(
            HOSPITALS[
                hospital
            ]["model"]
        )

        for hospital in HOSPITALS
    )

    all_hospital_datasets_available = all(

        file_exists(
            HOSPITALS[
                hospital
            ]["dataset"]
        )

        for hospital in HOSPITALS
    )

    all_hospital_scalers_available = all(

        file_exists(
            HOSPITALS[
                hospital
            ]["scaler"]
        )

        for hospital in HOSPITALS
    )

    print(
        "Hospital models available:",
        all_hospital_models_available
    )

    print(
        "Hospital datasets available:",
        all_hospital_datasets_available
    )

    print(
        "Hospital scalers available:",
        all_hospital_scalers_available
    )

    print("-" * 70)

    # --------------------------------------------------------
    # ACTUAL ADAPTIVE AGGREGATION
    # --------------------------------------------------------

    if all_hospital_models_available:

        try:

            aggregation_result = (
                perform_adaptive_aggregation()
            )

            if aggregation_result.get(
                "success"
            ):

                print(
                    "Adaptive aggregation "
                    "completed at startup."
                )

                print(
                    "Startup adaptive weights:"
                )

                for hospital in HOSPITALS:

                    print(
                        f"  {hospital}: "
                        f"{aggregation_result['weights'].get(hospital, 0.0):.2f}%"
                    )

            else:

                print(
                    "Adaptive aggregation "
                    "could not be completed."
                )

                print(
                    aggregation_result.get(
                        "message",
                        "Unknown aggregation error."
                    )
                )

        except Exception as error:

            print(
                "STARTUP AGGREGATION ERROR:",
                str(error)
            )

    else:

        print(
            "Not all hospital models are available."
        )

        print(
            "Using existing global model "
            "as fallback."
        )

    print("-" * 70)

    # --------------------------------------------------------
    # LOAD ACTIVE GLOBAL MODEL
    # --------------------------------------------------------

    initialize_global_model()

    print("-" * 70)

    print(
        "Final active model:",
        ACTIVE_MODEL_TYPE
    )

    print(
        "Final model status:",
        MODEL_STATUS
    )

    print(
        "Prediction history file:",
        RESULTS_FILE
    )

    print("=" * 70)


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    initialize_application()

    print(
        "Opening Microsoft Edge..."
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("=" * 70)

    threading.Timer(
        1.5,
        open_browser
    ).start()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )