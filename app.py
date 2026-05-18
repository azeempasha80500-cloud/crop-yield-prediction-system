from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import joblib
import os
import json

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

MODEL_FILE = "crop_model.pkl"
ENCODER_FILE = "encoders.pkl"

global_df = None

# =========================================================
# HTML TEMPLATE
# =========================================================

HTML = '''

<!DOCTYPE html>
<html lang="en">

<head>

    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Crop Yield Prediction</title>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">

    <style>

        body{
            background: linear-gradient(135deg,#d4fc79,#96e6a1);
            min-height:100vh;
            font-family: Arial, sans-serif;
            padding:40px 15px;
        }

        .main-container{
            max-width:1200px;
            margin:auto;
        }

        .card-box{
            background:white;
            padding:30px;
            border-radius:25px;
            box-shadow:0px 15px 35px rgba(0,0,0,0.15);
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
        }

        .btn-custom{
            width:100%;
            padding:14px;
            border:none;
            border-radius:12px;
            background:#198754;
            color:white;
            font-size:18px;
            font-weight:bold;
            transition:0.3s;
        }

        .btn-custom:hover{
            background:#157347;
            transform:translateY(-3px);
        }

        .result{
            margin-top:25px;
            background:#d1e7dd;
            padding:20px;
            border-radius:15px;
            text-align:center;
            animation:popUp 0.5s ease;
        }

        .chart-box{
            margin-top:30px;
            background:white;
            padding:25px;
            border-radius:20px;
            box-shadow:0px 10px 25px rgba(0,0,0,0.1);
        }

        canvas{
            max-height:450px;
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

<div class="main-container">

    <div class="card-box">

        <h1>Crop Yield Prediction</h1>

        <!-- Upload Dataset -->

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

        <hr class="my-4">

        <!-- Prediction Form -->

        <form action="/predict" method="POST">

            <div class="row">

                <div class="col-md-6 mb-3">
                    <input type="text"
                           name="state"
                           class="form-control"
                           placeholder="Enter State"
                           required>
                </div>

                <div class="col-md-6 mb-3">
                    <input type="text"
                           name="crop"
                           class="form-control"
                           placeholder="Enter Crop"
                           required>
                </div>

                <div class="col-md-6 mb-3">
                    <input type="number"
                           name="year"
                           class="form-control"
                           placeholder="Enter Year"
                           required>
                </div>

                <div class="col-md-6 mb-3">
                    <input type="number"
                           step="0.01"
                           name="area"
                           class="form-control"
                           placeholder="Enter Area"
                           required>
                </div>

            </div>

            <button type="submit" class="btn-custom">
                Predict Yield
            </button>

        </form>

        {% if prediction %}

        <div class="result">

            <h2>Predicted Yield</h2>

            <h3>{{ prediction }} Kg/Hectare</h3>

        </div>

        {% endif %}

        {% if message %}

        <div class="result">

            <h3>{{ message }}</h3>

        </div>

        {% endif %}

    </div>

    <!-- CHARTS -->

    {% if chart_labels %}

    <div class="chart-box">

        <h3 class="text-center mb-4">
            State Wise Crop Distribution
        </h3>

        <canvas id="cropChart"></canvas>

    </div>

    {% endif %}

</div>

<script>

    const labels = {{ chart_labels|safe if chart_labels else '[]' }};
    const values = {{ chart_values|safe if chart_values else '[]' }};

    if(labels.length > 0){

        const ctx = document.getElementById('cropChart');

        new Chart(ctx, {

            type: 'bar',

            data: {

                labels: labels,

                datasets: [{

                    label: 'Number of Crops',

                    data: values,

                    borderWidth: 2

                }]

            },

            options: {

                responsive: true,

                plugins: {

                    legend: {

                        display: true

                    }

                }

            }

        });

    }

</script>

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
# UPLOAD DATASET
# =========================================================

@app.route('/upload', methods=['POST'])
def upload():

    global global_df

    try:

        file = request.files['dataset']

        if not file:

            return render_template_string(
                HTML,
                message="No file uploaded"
            )

        # Read CSV
        df = pd.read_csv(file)

        global_df = df.copy()

        # Clean columns
        df.columns = df.columns.str.strip()

        # Rename columns
        df = df.rename(columns={
            "Crop Name": "Crop",
            "Area (UOM:Ha(Hectare)), Scaling Factor:1000": "Area",
            "Yield (UOM:Kg/Ha(KilogramperHectare)), Scaling Factor:1": "Yield"
        })

        # Drop unwanted columns
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

        df = df.dropna()

        # Remove outliers
        df = df[df["Yield"] < df["Yield"].quantile(0.97)]

        # Label encoding
        state_encoder = LabelEncoder()
        crop_encoder = LabelEncoder()

        df["State"] = state_encoder.fit_transform(df["State"])

        df["Crop"] = crop_encoder.fit_transform(df["Crop"])

        # Features
        X = df[["State", "Crop", "Year", "Area"]]

        # Target
        y = df["Yield"]

        # Train split
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

        # Train
        model.fit(X_train, y_train)

        # Save model
        joblib.dump(model, MODEL_FILE)

        # Save encoders
        joblib.dump({
            "state_encoder": state_encoder,
            "crop_encoder": crop_encoder
        }, ENCODER_FILE)

        # =========================
        # CHART DATA
        # =========================

        chart_data = global_df.groupby("State")["Crop Name"].count().sort_values(ascending=False).head(10)

        chart_labels = list(chart_data.index)

        chart_values = list(chart_data.values)

        return render_template_string(
            HTML,
            message="Dataset Uploaded & Model Trained Successfully",
            chart_labels=json.dumps(chart_labels),
            chart_values=json.dumps(chart_values)
        )

    except Exception as e:

        return render_template_string(
            HTML,
            message=f"Upload Error: {str(e)}"
        )

# =========================================================
# PREDICTION
# =========================================================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        if not os.path.exists(MODEL_FILE):

            return render_template_string(
                HTML,
                message="Please Upload Dataset First"
            )

        # Load model
        model = joblib.load(MODEL_FILE)

        encoders = joblib.load(ENCODER_FILE)

        state_encoder = encoders["state_encoder"]

        crop_encoder = encoders["crop_encoder"]

        # Inputs
        state = request.form['state'].strip()

        crop = request.form['crop'].strip()

        year = int(request.form['year'])

        area = float(request.form['area'])

        # Check state
        if state not in state_encoder.classes_:

            return render_template_string(
                HTML,
                message=f"State '{state}' not found"
            )

        # Check crop
        if crop not in crop_encoder.classes_:

            return render_template_string(
                HTML,
                message=f"Crop '{crop}' not found"
            )

        # Encode
        state_encoded = state_encoder.transform([state])[0]

        crop_encoded = crop_encoder.transform([crop])[0]

        # Features
        features = np.array([
            [state_encoded, crop_encoded, year, area]
        ])

        # Predict
        prediction = model.predict(features)[0]

        # Mini Map Data
        chart_data = global_df.groupby("State")["Crop Name"].count().sort_values(ascending=False).head(10)

        chart_labels = list(chart_data.index)

        chart_values = list(chart_data.values)

        return render_template_string(
            HTML,
            prediction=round(float(prediction), 2),
            chart_labels=json.dumps(chart_labels),
            chart_values=json.dumps(chart_values)
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
