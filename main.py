from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURES = [
    ("Radius Mean", "radius_mean"), ("Texture Mean", "texture_mean"),
    ("Perimeter Mean", "perimeter_mean"), ("Area Mean", "area_mean"),
    ("Smoothness Mean", "smoothness_mean"), ("Compactness Mean", "compactness_mean"),
    ("Concavity Mean", "concavity_mean"), ("Concave Points Mean", "concave_pts_mean"),
    ("Symmetry Mean", "symmetry_mean"), ("Fractal Dim Mean", "fractal_dim_mean"),
    ("Radius SE", "radius_se"), ("Texture SE", "texture_se"),
    ("Perimeter SE", "perimeter_se"), ("Area SE", "area_se"),
    ("Smoothness SE", "smoothness_se"), ("Compactness SE", "compactness_se"),
    ("Concavity SE", "concavity_se"), ("Concave Points SE", "concave_pts_se"),
    ("Symmetry SE", "symmetry_se"), ("Fractal Dim SE", "fractal_dim_se"),
    ("Radius Worst", "radius_worst"), ("Texture Worst", "texture_worst"),
    ("Perimeter Worst", "perimeter_worst"), ("Area Worst", "area_worst"),
    ("Smoothness Worst", "smoothness_worst"), ("Compactness Worst", "compactness_worst"),
    ("Concavity Worst", "concavity_worst"), ("Concave Points Worst", "concave_pts_worst"),
    ("Symmetry Worst", "symmetry_worst"), ("Fractal Dim Worst", "fractal_dim_worst"),
]

def build_form(result=None, confidence=None):
    inputs_mean = ""
    inputs_se = ""
    inputs_worst = ""

    for i, (label, name) in enumerate(FEATURES):
        field = f"""
        <div class="field">
            <label>{label}</label>
            <input type="number" name="{name}" step="any" required placeholder="0.00">
        </div>"""
        if i < 10:
            inputs_mean += field
        elif i < 20:
            inputs_se += field
        else:
            inputs_worst += field

    result_html = ""
    if result:
        if result == "Benign":
            box_class = "result-benign"
            text_class = "benign-text"
        else:
            box_class = "result-malignant"
            text_class = "malignant-text"
        conf_html = f'<div class="confidence">Confidence: {confidence}%</div>' if confidence else ""
        result_html = f"""
        <div class="result-box {box_class}">
            <div class="result-label">Prediction Result</div>
            <div class="result-value {text_class}">{result}</div>
            {conf_html}
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Breast Cancer Predictor</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; padding: 40px 20px; }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        header {{ text-align: center; margin-bottom: 40px; }}
        header h1 {{ font-size: 2rem; color: #a78bfa; }}
        header p {{ color: #94a3b8; margin-top: 8px; }}
        .badge {{ display: inline-block; background: #1e1b4b; border: 1px solid #7c3aed; color: #a78bfa; padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; margin-top: 10px; }}
        .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 30px; margin-bottom: 24px; }}
        .card h2 {{ font-size: 1rem; color: #a78bfa; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid #334155; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 16px; }}
        .field label {{ display: block; font-size: 0.78rem; color: #94a3b8; margin-bottom: 5px; text-transform: uppercase; }}
        .field input {{ width: 100%; padding: 9px 12px; background: #0f172a; border: 1px solid #334155; border-radius: 6px; color: #e2e8f0; font-size: 0.9rem; }}
        .field input:focus {{ outline: none; border-color: #7c3aed; }}
        .submit-btn {{ width: 100%; padding: 14px; background: #7c3aed; color: white; border: none; border-radius: 8px; font-size: 1rem; font-weight: 600; cursor: pointer; margin-top: 10px; }}
        .submit-btn:hover {{ background: #6d28d9; }}
        .result-box {{ text-align: center; padding: 30px; border-radius: 12px; margin-bottom: 24px; border: 2px solid; }}
        .result-benign {{ background: #052e16; border-color: #16a34a; }}
        .result-malignant {{ background: #2d0b0b; border-color: #dc2626; }}
        .result-label {{ font-size: 0.85rem; text-transform: uppercase; color: #94a3b8; margin-bottom: 8px; }}
        .result-value {{ font-size: 2.2rem; font-weight: 700; }}
        .benign-text {{ color: #4ade80; }}
        .malignant-text {{ color: #f87171; }}
        .confidence {{ font-size: 0.9rem; color: #94a3b8; margin-top: 8px; }}
        .footer {{ text-align: center; margin-top: 40px; color: #475569; font-size: 0.8rem; }}
        .footer a {{ color: #7c3aed; text-decoration: none; }}
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>Breast Cancer Prediction</h1>
        <p>Enter tumor measurements to predict malignancy</p>
        <span class="badge">Model Accuracy: 96.5%</span>
    </header>
    {result_html}
    <form method="post" action="/predict">
        <div class="card">
            <h2>Mean Values</h2>
            <div class="grid">{inputs_mean}</div>
        </div>
        <div class="card">
            <h2>Standard Error Values</h2>
            <div class="grid">{inputs_se}</div>
        </div>
        <div class="card">
            <h2>Worst Values</h2>
            <div class="grid">{inputs_worst}</div>
        </div>
        <button type="submit" class="submit-btn">Run Prediction</button>
    </form>
    <div class="footer">
        Built by <a href="https://github.com/abdulrafay1942" target="_blank">Abdul Rafay Chaudhary</a> &nbsp;|&nbsp;
        <a href="https://www.linkedin.com/in/abdul-rafay-04397332a" target="_blank">LinkedIn</a> &nbsp;|&nbsp;
        <a href="https://www.kaggle.com/abdulrafay1942" target="_blank">Kaggle</a>
    </div>
</div>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=build_form())

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form = await request.form()
    try:
        values = [float(form.get(name)) for _, name in FEATURES]
        scaled = scaler.transform([values])
        prediction = model.predict(scaled)[0]
        confidence = round(max(model.predict_proba(scaled)[0]) * 100, 2)
        result = "Malignant" if prediction == 1 else "Benign"
    except Exception as e:
        result = f"Error: {str(e)}"
        confidence = None
    return HTMLResponse(content=build_form(result, confidence))