from flask import Flask, request, jsonify
import pandas as pd
import joblib
import re

# Load the trained pipeline
pipeline = joblib.load("decision_tree_pipeline.pkl")

# Column names (must match training order)
numerical_columns = ["age", "income", "net_worth", "monthly_expense", "debt_percent", "savings_percent"]

app = Flask(__name__)
@app.route('/')
def home():
    return "Flask API is running successfully!"

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse incoming JSON
        #data = request.get_json()
        data = request.get_json(force=True)
        required_fields = ["age", "income", "net_worth", "monthly_expense", "debt_percent", "savings_percent"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: ee fields ni fill cheyyi ra labbey {','.join(missing_fields)}"}), 400
        # #email pattern check
        # email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        # if not re.match(email_pattern,data["email"]):
        #     return jsonify({"error":"Inavlid email format bhayya"}),400
        # Numeric validations
        if data["income"] < 0:
            return jsonify({"error": "Income negative ela isthav ra"}), 400
        if data["monthly_expense"] < 0:
            return jsonify({"error": "Monthly expenses negative ela isthav ra loveday"}), 400
        if not (0 <= data["debt_percent"] <= 100):
            return jsonify({"error": "Debt percent must be between 0 and 100"}), 400
        if not (0 <= data["savings_percent"] <= 100):
            return jsonify({"error": "Savings percent must be between 0 and 100"}), 400
        # Minimum criteria
        # if data["income"] < 5000:
        #     return jsonify({"error": "Income must be at least 5000"}), 400
        if data["age"] < 18:
            return jsonify({"error": "18 years lekunda neeku salary osthey ichhina vaadinii, ninnu iddarini policulaki pattistha, calling 100"}), 400
        #user = User(**data).save()
        #return jsonify({"message": "User created successfully", "id": str(user.id)}), 201
    

        # Ensure it's a list of dicts or single dict
        if isinstance(data, dict):
            data = [data]  # wrap single record in a list

        # Convert to DataFrame
        input_df = pd.DataFrame(data, columns=numerical_columns)

        # Get scaled features
        scaled_features = pipeline.named_steps["scaler"].transform(input_df)

        # Get predictions
        predictions = pipeline.predict(input_df)

        # Build response
        # results = []
        # for i in range(len(input_df)):
        #     results.append({
        #         "features": input_df.iloc[i].tolist(),
        #         "scaledFeatures": scaled_features[i].tolist(),
        #         "prediction": (predictions[i])
        #     })

        # return jsonify(results["prediction"])
        result = {
            "features": input_df.iloc[0].tolist(),
            "scaledFeatures": scaled_features[0].tolist(),
            "prediction": predictions[0]
            }
        return jsonify(result["prediction"])
  

    except Exception as e:
        return jsonify({"error": str(e)}), 400



if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=5000)
    app.run(debug=True)
    
