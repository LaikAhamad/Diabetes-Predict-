from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
encoder = joblib.load("onehot_encoder.joblib")

numeric_cols = ["Age", "BMI", "Glucose"]
categorical_cols = ["Gender",  "smaker", "FamilyHistory"]

@app.route("/", methods=["GET","POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            X_num = np.array([
                float(request.form["Age"]),
                float(request.form["BMI"]),
                float(request.form["Glucose"])
            ]).reshape(1, -1)

        #categorical inputs
            X_cat = np.array([[
                request.form["Gender"],
                request.form["smoker"],
                request.form["FamilyHistory"]
            ]])

        #Apply preprocessing
            X_num_scaled = scaler.transform(X_num)
            x_cat_encoded = encoder.transform(X_cat)
            x_final = np.hstack([X_num_scaled, x_cat_encoded])

            pred = model.predict(x_final)[0]
            result = "Diabetic" if pred == 1 else "Not Diabetic"
        
        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
        
