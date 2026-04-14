from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model
def load_models():
    try:
        model = joblib.load("model/model_bmw.pkl")
        le_model = joblib.load("model/le_model.pkl")
        le_trans = joblib.load("model/le_trans.pkl")
        le_fuel = joblib.load("model/le_fuel.pkl")
        return model, le_model, le_trans, le_fuel
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None

model, le_model, le_trans, le_fuel = load_models()

# Konfigurasi
CURRENT_YEAR = 2026
USD_TO_GBP = 0.79
GBP_TO_IDR = 22500

@app.route('/', methods=['GET', 'POST'])
def index():
    if model is None:
        return "Model tidak dapat dimuat", 500
    
    harga_gbp = None
    harga_idr = None
    error = None
    
    # Get unique values for dropdowns
    models_list = le_model.classes_.tolist() if hasattr(le_model, 'classes_') else []
    transmissions_list = le_trans.classes_.tolist() if hasattr(le_trans, 'classes_') else []
    fuels_list = le_fuel.classes_.tolist() if hasattr(le_fuel, 'classes_') else []
    
    if request.method == 'POST':
        try:
            # Get form data
            model_name = request.form['model']
            year = int(request.form['year'])
            transmission = request.form['transmission']
            mileage = int(request.form['mileage'])
            fuel = request.form['fuel']
            tax = int(request.form['tax'])
            mpg = float(request.form['mpg'])
            engine = float(request.form['engine'])
            
            # Encoding
            model_enc = le_model.transform([model_name])[0]
            trans_enc = le_trans.transform([transmission])[0]
            fuel_enc = le_fuel.transform([fuel])[0]
            
            # Predict
            data = np.array([[
                model_enc, year, trans_enc, mileage, 
                fuel_enc, tax, mpg, engine
            ]])
            
            price_usd = model.predict(data)[0]
            
            # Apply business logic
            age = CURRENT_YEAR - year
            if age > 0:
                price_usd *= (1 - 0.05) ** age
            
            if mileage > 100000:
                price_usd *= 0.90
            
            if engine < 2.0:
                price_usd *= 0.93
            
            # Convert currency
            harga_gbp = round(price_usd * USD_TO_GBP, 2)
            harga_idr = round(harga_gbp * GBP_TO_IDR)
            
        except Exception as e:
            error = f"Error: {str(e)}"
    
    return render_template('index.html', 
                         models=models_list,
                         transmissions=transmissions_list,
                         fuels=fuels_list,
                         harga_gbp=harga_gbp,
                         harga_idr=harga_idr,
                         error=error)

if __name__ == '__main__':
    app.run(debug=True)