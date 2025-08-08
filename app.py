import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load base DataFrame and model
df_1 = pd.read_csv("churn.csv")  # Your reference dummy-free dataset
model = pickle.load(open("modelPredic.sav", "rb"))

# Load the expected feature columns used during model training
expected_cols = pickle.load(open("features.pkl", "rb"))  # Save this during training!

@app.route("/")
def loadPage():
    return render_template("home.html")

@app.route("/", methods=["POST"])
def predict():
    try:
        # Read form inputs
        input_data = [request.form[f"query{i}"] for i in range(1, 20)]

        # Create DataFrame from input
        new_df = pd.DataFrame([input_data], columns=[
            'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
            'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'tenure'
        ])

        # Convert appropriate columns
        new_df['SeniorCitizen'] = new_df['SeniorCitizen'].astype(int)
        new_df['MonthlyCharges'] = new_df['MonthlyCharges'].astype(float)
        new_df['TotalCharges'] = new_df['TotalCharges'].astype(float)
        new_df['tenure'] = new_df['tenure'].astype(int)

        # Add new row to df_1 for consistent processing
        df_2 = pd.concat([df_1, new_df], ignore_index=True)

        # Create tenure_group column
        labels = [f"{i} - {i+11}" for i in range(1, 72, 12)]
        df_2['tenure_group'] = pd.cut(df_2['tenure'], range(1, 80, 12), right=False, labels=labels)
        df_2.drop(columns=['tenure'], inplace=True)

        # One-hot encode the necessary columns
        encoded_df = pd.get_dummies(df_2[[
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group'
        ]])

        # Fill missing columns
        for col in expected_cols:
            if col not in encoded_df.columns:
                encoded_df[col] = 0

        # Reorder columns
        final_df = encoded_df[expected_cols]

        # Predict the last row
        pred = model.predict(final_df.tail(1))[0]
        prob = model.predict_proba(final_df.tail(1))[0][1]

        result = "This customer is likely to churn!" if pred == 1 else "This customer is likely to stay."
        confidence = f"Confidence: {prob * 100:.2f}%"

        return render_template("home.html", output1=result, output2=confidence)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    import webbrowser
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)
