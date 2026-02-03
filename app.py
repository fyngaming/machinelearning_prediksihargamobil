from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# ===============================
# PATH AMAN
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

model = joblib.load(os.path.join(MODEL_DIR, "model_bmw.pkl"))
le_model = joblib.load(os.path.join(MODEL_DIR, "le_model.pkl"))
le_trans = joblib.load(os.path.join(MODEL_DIR, "le_trans.pkl"))
le_fuel = joblib.load(os.path.join(MODEL_DIR, "le_fuel.pkl"))

# ===============================
# KONFIGURASI
# ===============================
CURRENT_YEAR = 2026
USD_TO_GBP = 0.79
GBP_TO_IDR = 22500

# ===============================
# ROUTE
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    harga_gbp = None
    harga_idr = None
    error = None

    models = le_model.classes_
    transmissions = le_trans.classes_
    fuels = le_fuel.classes_

    if request.method == "POST":
        try:
            model_name = request.form["model"]
            year = int(request.form["year"])
            transmission = request.form["transmission"]
            mileage = int(request.form["mileage"])
            fuel = request.form["fuel"]
            tax = int(request.form["tax"])
            mpg = float(request.form["mpg"])
            engine = float(request.form["engine"])

            # ENCODING
            model_enc = le_model.transform([model_name])[0]
            trans_enc = le_trans.transform([transmission])[0]
            fuel_enc = le_fuel.transform([fuel])[0]

            # DATA (8 FITUR – SAMA DENGAN TRAINING)
            data = np.array([[ 
                model_enc,
                year,
                trans_enc,
                mileage,
                fuel_enc,
                tax,
                mpg,
                engine
            ]])

            # PREDIKSI (USD)
            price_usd = model.predict(data)[0]

            # ===============================
            # LOGIKA EKONOMI REALISTIS
            # ===============================
            age = CURRENT_YEAR - year
            if age > 0:
                price_usd *= (1 - 0.05) ** age  # depresiasi 5% / tahun

            if mileage > 100000:
                price_usd *= 0.90

            if engine < 2.0:
                price_usd *= 0.93

            # ===============================
            # KONVERSI MATA UANG
            # ===============================
            harga_gbp = round(price_usd * USD_TO_GBP, 2)
            harga_idr = round(harga_gbp * GBP_TO_IDR)

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        models=models,
        transmissions=transmissions,
        fuels=fuels,
        harga_gbp=harga_gbp,
        harga_idr=harga_idr,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
