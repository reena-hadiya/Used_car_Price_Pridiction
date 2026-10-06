import joblib
import pandas as pd
import os


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

preprocessor_path = os.path.join(
    BASE_DIR,
    "models",
    "scaler_pipeline.pkl"
)

model_path = os.path.join(
    BASE_DIR,
    "models",
    "car_price_model.pkl"
)


preprocessor = joblib.load(preprocessor_path)

model = joblib.load(model_path)


sample_car = pd.DataFrame([{
    'distance_travelled(kms)': 35000,
    'car_age': 3,
    'km_per_year': 11666,
    'brand_rank': 1,
    'fuel_type': 'Petrol',
    'brand': 'Hyundai',
    'model_name': 'i20',
    'city': 'Mumbai',
    'new and less used': 0
}])


print("\nINPUT:")
print(sample_car)


processed = preprocessor.transform(sample_car)


print("\nPROCESSED:")
print(processed)


prediction = model.predict(processed)[0]


print("\nPREDICTED PRICE:")
print(f"₹{prediction:,.2f}")