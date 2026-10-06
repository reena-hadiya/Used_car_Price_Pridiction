from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FRONTEND_FOLDER = os.path.join(BASE_DIR, "frontend")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")

app = Flask(
    __name__,
    template_folder=FRONTEND_FOLDER,
    static_folder=FRONTEND_FOLDER,
    static_url_path="/static"
)

PREPROCESSOR_PATH = os.path.join(
    MODEL_FOLDER,
    "scaler_pipeline.pkl"
)

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "car_price_model.pkl"
)


preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)

brand_models = {

    "Honda": [
        "Brio",
        "City",
        "Jazz",
        "WR-V",
        "Civic",
        "Amaze",
        "CR-V",
        "Accord",
        "BR-V"
    ],

    "Nissan": [
        "Sunny",
        "Micra"
    ],

    "Toyota": [
        "Fortuner",
        "Innova",
        "Corolla",
        "Etios",
        "Yaris",
        "Camry",
        "Qualis"
    ],

    "Mercedes-Benz": [
        "E-Class",
        "GLE",
        "GLA",
        "C-Class",
        "S-Class",
        "CLA",
        "B-Class",
        "A-Class",
        "GLC",
        "GLS",
        "G-Class",
        "CLS",
        "M-Class"
    ],

    "Hyundai": [
        "Verna",
        "i20",
        "Xcent",
        "Creta",
        "i10",
        "Elantra",
        "Santro",
        "Venue",
        "Tucson",
        "Eon"
    ],

    "Maruti Suzuki": [
        "Swift",
        "Baleno",
        "Ritz",
        "Celerio",
        "A-Star",
        "Ciaz",
        "Dzire",
        "Alto",
        "Wagon",
        "Ertiga",
        "S-Cross",
        "Ignis",
        "XL6"
    ],

    "Renault": [
        "Pulse",
        "Lodgy",
        "Duster",
        "Kwid",
        "Triber",
        "Captur"
    ],

    "Volkswagen": [
        "Polo",
        "Vento",
        "Passat",
        "Jetta",
        "Ameo",
        "Tiguan"
    ],

    "Skoda": [
        "Rapid",
        "Superb",
        "Octavia",
        "Laura",
        "Fabia",
        "Yeti"
    ],

    "BMW": [
        "X1",
        "Z4",
        "X3",
        "5",
        "3",
        "X5",
        "X7",
        "X4"
    ],

    "Tata": [
        "Nano",
        "Tiago",
        "Altroz",
        "Zest",
        "Tigor",
        "Nexon",
        "Safari",
        "Harrier",
        "Hexa",
        "Indica"
    ],

    "Audi": [
        "Q5",
        "A3",
        "A4",
        "Q3",
        "Q7",
        "A6",
        "A8"
    ],

    "Ford": [
        "Endeavour",
        "Figo",
        "EcoSport",
        "Mustang",
        "Fiesta",
        "Aspire",
        "Freestyle"
    ],

    "Mahindra": [
        "XUV500",
        "KUV100",
        "TUV300",
        "Bolero",
        "Scorpio",
        "Thar",
        "XUV300"
    ],

    "Jaguar": [
        "F-Pace",
        "XE",
        "XF",
        "XJ"
    ],

    "Kia": [
        "Seltos",
        "Sonet"
    ],

    "Jeep": [
        "Compass",
        "Jeep"
    ],

    "Volvo": [
        "V40",
        "S60",
        "XC90",
        "S90",
        "XC60"
    ],

    "MG": [
        "Hector"
    ],

    "Lexus": [
        "NX",
        "ES"
    ],

    "Mitsubishi": [
        "Pajero",
        "Outlander"
    ],

    "Fiat": [
        "Punto",
        "Avventura",
        "Abarth"
    ]

}


@app.route("/")
def home():

    return render_template(
        "index.html",
        brand_models=brand_models
    )

@app.route("/predict", methods=["POST"])
def predict():

    try:

        brand = request.form["brand"]

        model_name = request.form["model_name"]

        distance_travelled = float(
            request.form["distance_travelled"]
        )

        fuel_type = request.form["fuel_type"]

        city = request.form["city"]

        brand_rank = int(
            request.form["brand_rank"]
        )

        new_and_less_used = int(
            request.form["new_and_less_used"]
        )

        car_age = float(
            request.form["car_age"]
        )

        km_per_year = float(
            request.form["km_per_year"]
        )

        sample_car = pd.DataFrame([{

            "brand": brand,

            "model_name": model_name,

            "distance_travelled(kms)": distance_travelled,

            "fuel_type": fuel_type,

            "city": city,

            "brand_rank": brand_rank,

            "new and less used": new_and_less_used,

            "car_age": car_age,

            "km_per_year": km_per_year

        }])

        processed_data = preprocessor.transform(
            sample_car
        )

        predicted_price = model.predict(
            processed_data
        )[0]

        return render_template(
            "index.html",
            brand_models=brand_models,
            prediction=f"₹{predicted_price:,.2f}",
            selected_brand=brand,
            selected_model=model_name
        )


    except Exception as e:

        return render_template(
            "index.html",
            brand_models=brand_models,
            error=str(e)
        )

if __name__ == "__main__":

    app.run(
        debug=True
    )