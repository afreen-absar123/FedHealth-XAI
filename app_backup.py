from flask import Flask, render_template, request

from prediction import predict_heart_disease


app = Flask(__name__)


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# PATIENT PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        age = float(request.form["age"])
        sex = float(request.form["sex"])
        cp = float(request.form["cp"])
        trestbps = float(request.form["trestbps"])
        chol = float(request.form["chol"])
        fbs = float(request.form["fbs"])
        restecg = float(request.form["restecg"])
        thalach = float(request.form["thalach"])
        exang = float(request.form["exang"])
        oldpeak = float(request.form["oldpeak"])
        slope = float(request.form["slope"])
        ca = float(request.form["ca"])
        thal = float(request.form["thal"])


        patient_data = [

            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal

        ]


        prediction, probability = predict_heart_disease(
            patient_data
        )


        return render_template(
            "index.html",
            prediction=prediction,
            probability=f"{probability:.2f}"
        )


    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )


# ==========================================
# START FLASK
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)