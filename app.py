from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

MODEL_FILE = "crop_model.pkl"
ENCODER_FILE = "encoders.pkl"

# =========================================================
# HTML PAGE
# =========================================================

HTML = '''

<!DOCTYPE html>
<html lang="en">

<head>

    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Crop Yield Prediction</title>

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">

    <style>

        body{
            background: linear-gradient(135deg,#d4fc79,#96e6a1);
            min-height:100vh;
            display:flex;
            justify-content:center;
            align-items:center;
            font-family:Arial;
            padding:20px;
        }

        .container-box{
            width:700px;
            background:white;
            padding:40px;
            border-radius:25px;
            box-shadow:0px 15px 35px rgba(0,0,0,0.2);
            animation:fadeIn 1s ease;
        }

        h1{
            text-align:center;
            color:#198754;
            font-weight:bold;
            margin-bottom:30px;
            animation:slideDown 1s ease;
        }

        .form-control{
            padding:14px;
            border-radius:12px;
            transition:0.3s;
        }

        .form-control:focus{
            transform:scale(1.02);
            box-shadow:0 0 10px rgba(25,135,84,0.3);
        }

        .btn-custom{
            width:100%;
            padding:14px;
            border:none;
            border-radius:14px;
            background:#198754;
            color:white;
            font-size:18px;
            font-weight:bold;
            transition:0.3s;
        }

        .btn-custom:hover{
            transform:translateY(-3px);
            background:#157347;
        }

        .result{
            margin-top:30px;
            padding:20px;
            background:#d1e7dd;
            border-radius:15px;
            text-align:center;
            animation:popUp 0.6s ease;
        }

        .result h2{
            color:#0f5132;
            font-weight:bold;
        }

        .upload-box{
            margin-bottom:25px;
            padding:20px;
            border:2px dashed #198754;
            border-radius:15px;
            background:#f8f9fa;
        }

        .footer{
            text-align:center;
            margin-top:20px;
            color:gray;
        }

        @keyframes fadeIn{
            from{
                opacity:0;
                transform:translateY(20px);
            }
            to{
                opacity:1;
                transform:translateY(0);
            }
        }

        @keyframes slideDown{
            from{
                opacity:0;
                transform:translateY(-30px);
            }
            to{
                opacity:1;
                transform:translateY(0);
            }
        }

        @keyframes popUp{
            from{
                opacity:0;
                transform:scale(0.7);
            }
            to{
                opacity:1;
                transform:scale(1);
            }
        }

    </style>

</head>

<body>

<div class="container-box">

    <h1>Crop Yield Prediction</h1>

    <!-- Upload Dataset -->

    <div class="upload-box">

        <form action="/upload" method="POST" enctype="multipart/form-data">

            <div class="mb-3">

                <label class="fw-bold">
                    Upload Dataset CSV
                </label>

                <input type="file"
                       name="dataset"
                       class="form-control"
                       required>

            </div>

            <button type="submit" class="btn-custom">
                Upload & Train Model
            </button>

        </form>

    </div>

    <!-- Prediction Form -->

    <form action="/predict" method="POST">

        <div class="mb-3">
            <input type="text"
                   name="state"
                   class="form-control"
                   placeholder="Enter State"
                   required>
        </div>

        <div class="mb-3">
            <input type="text"
                   name="crop"
                   class="form-control"
                   placeholder="Enter Crop"
                   required>
        </div>

        <div class="mb-3">
            <input type="number"
                   name="year"
                   class="form-control"
                   placeholder="Enter Year"
                   required>
        </div>

        <div class="mb-3">
            <input type="number"
                   step="0.01"
                   name="area"
                   class="form-control"
                   placeholder="Enter Area"
                   required>
        </div>

        <button type="submit" class="btn-custom">
            Predict Yield
        </button>

    </form>

    <!-- Prediction Result -->

    {% if prediction %}

    <div class="result">

        <h2>Predicted Yield</h2>

        <h3>{{ prediction }} Kg/Hectare</h3>

    </div>

    {% endif %}

    <!-- Message -->

    {% if message %}

    <div class="result">

        <h3>{{ message }}</h3>

    </div>

    {% endif %}

    <div class="footer">
        Machine Learning Based Agriculture Prediction System
    </div>

</div>

</body>
</html>

'''

# =========================================================
# HOME
# =========================================================

@app.route('/')
def home():

    return render_template_string(HTML)

# =========================================================
# UPLOAD DATASET & TRAIN MODEL
# =========================================================

@app.route('/upload', methods=['POST'])
def upload():

    try:

        file = request.files['dataset']

        if not file:

            return render_template_string(
                HTML,
                message="No file uploaded"
            )

        # Read CSV
        df = pd.read_csv(file)

        # Clean column names
        df.columns = df.columns.str.strip()

        # Rename columns
        df = df.rename(columns={
            "Crop Name": "Crop",
            "Area (UOM:Ha(Hectare)), Scaling Factor:1000": "Area",
            "Yield (UOM:Kg/Ha(KilogramperHectare)), Scaling Factor:1": "Yield"
        })

        # Drop unnecessary columns
        drop_cols = [
            "Country",
            "Additional Info",
            "Area Percentage To All India (%) (UOM:%(Percentage)), Scaling Factor:1",
            "Production",
            "Production Percentage To All India (%) (UOM:%(Percentage)), Scaling Factor:1"
        ]

        df = df.drop(columns=drop_cols, errors='ignore')

        # Clean year
        df["Year"] = df["Year"].astype(str).str.extract(r'(\\d{4})')

        df = df.dropna(subset=["Year"])

        df["Year"] = df["Year"].astype(int)

        # Remove nulls
        df = df.dropna()

        # Remove outliers
        df = df[df["Yield"] < df["Yield"].quantile(0.97)]

        # Label Encoding
        state_encoder = LabelEncoder()
        crop_encoder = LabelEncoder()

        df["State"] = state_encoder.fit_transform(df["State"])

        df["Crop"] = crop_encoder.fit_transform(df["Crop"])

        # Features
        X = df[["State", "Crop", "Year", "Area"]]

        # Target
        y = df["Yield"]

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42
        )

        # Train model
        model.fit(X_train, y_train)

        # Save model
        joblib.dump(model, MODEL_FILE)

        # Save encoders
        joblib.dump({
            "state_encoder": state_encoder,
            "crop_encoder": crop_encoder
        }, ENCODER_FILE)

        return render_template_string(
            HTML,
            message="Dataset Uploaded & Model Trained Successfully"
        )

    except Exception as e:

        return render_template_string(
            HTML,
            message=f"Upload Error: {str(e)}"
        )

# =========================================================
# PREDICT
# =========================================================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        # Check model exists
        if not os.path.exists(MODEL_FILE):

            return render_template_string(
                HTML,
                message="Please Upload Dataset First"
            )

        # Load model
        model = joblib.load(MODEL_FILE)

        # Load encoders
        encoders = joblib.load(ENCODER_FILE)

        state_encoder = encoders["state_encoder"]
        crop_encoder = encoders["crop_encoder"]

        # Get values
        state = request.form['state'].strip()
        crop = request.form['crop'].strip()

        year = int(request.form['year'])

        area = float(request.form['area'])

        # Check state exists
        if state not in state_encoder.classes_:

            return render_template_string(
                HTML,
                message=f"State '{state}' not found in dataset"
            )

        # Check crop exists
        if crop not in crop_encoder.classes_:

            return render_template_string(
                HTML,
                message=f"Crop '{crop}' not found in dataset"
            )

        # Encode values
        state_encoded = state_encoder.transform([state])[0]

        crop_encoded = crop_encoder.transform([crop])[0]

        # Features
        features = np.array([
            [state_encoded, crop_encoded, year, area]
        ])

        # Prediction
        prediction = model.predict(features)[0]

        return render_template_string(
            HTML,
            prediction=round(float(prediction), 2)
        )

    except Exception as e:

        return render_template_string(
            HTML,
            message=f"Prediction Error: {str(e)}"
        )

# =========================================================
# MAIN
# =========================================================

if __name__ == '__main__':

    app.run(debug=True)
