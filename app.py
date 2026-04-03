from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)


MODEL_PATH   = "car_price_model.pkl"
COLUMNS_PATH = "model_columns.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
    raise FileNotFoundError(
        "Model files not found!\n"
        "Please run  python model_training.py  first to train and save the model."
    )

model          = joblib.load(MODEL_PATH)
model_columns  = joblib.load(COLUMNS_PATH)   



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()


        input_df = pd.DataFrame([{
            "make_year"          : int(data["make_year"]),
            "mileage_kmpl"       : float(data["mileage_kmpl"]),
            "engine_cc"          : int(data["engine_cc"]),
            "fuel_type"          : data["fuel_type"],          
            "owner_count"        : int(data["owner_count"]),
            "brand"              : data["brand"],
            "transmission"       : data["transmission"],       
            "color"              : data["color"],
            "service_history"    : data["service_history"],    
            "accidents_reported" : int(data["accidents_reported"]),
            "insurance_valid"    : data["insurance_valid"],    
        }])


        input_encoded = pd.get_dummies(input_df, drop_first=True)


        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)


        predicted_price = model.predict(input_encoded)[0]

        return jsonify({
            "success"        : True,
            "predicted_price": round(float(predicted_price), 2)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400



if __name__ == "__main__":
    app.run(debug=True, port=5000)
