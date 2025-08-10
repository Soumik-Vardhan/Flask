from flask import Flask, request, jsonify
import pandas as pd
import joblib

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
        data = request.get_json()

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
    app.run(host="0.0.0.0", port=5000)
    #app.run(debug=True)
    
