# app.py
import os
import requests
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd

# -------- CONFIG --------
UPLOAD_DIR = os.path.join("static", "uploads")
MODEL_FILENAME = "plant_disease_model_1_latest.pt"
MODEL_PATH = os.path.join(".", MODEL_FILENAME)
# ------------------------

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Attempt to download model if MODEL_URL env var is provided and model not present
MODEL_URL = os.environ.get("MODEL_URL")
if not os.path.exists(MODEL_PATH) and MODEL_URL:
    print("Model not found locally — downloading from MODEL_URL ...")
    r = requests.get(MODEL_URL, stream=True)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model downloaded.")

# Load CSVs
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load model (map to CPU)
model = CNN.CNN(39)
state = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(state)
model.eval()

def prediction(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)             # 0..1 float tensor CxHxW
    input_data = input_data.view((-1, 3, 224, 224))
    with torch.no_grad():
        output = model(input_data)
    output = output.cpu().numpy()
    index = int(np.argmax(output))
    return index

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part", 400
        image = request.files['image']
        if image.filename == '':
            return "No selected file", 400
        filename = secure_filename(image.filename)
        file_path = os.path.join(UPLOAD_DIR, filename)
        image.save(file_path)

        pred = prediction(file_path)
        title = disease_info['disease_name'].iloc[pred]
        description = disease_info['description'].iloc[pred]
        prevent = disease_info['Possible Steps'].iloc[pred]
        image_url = disease_info['image_url'].iloc[pred]
        supplement_name = supplement_info['supplement name'].iloc[pred]
        supplement_image_url = supplement_info['supplement image'].iloc[pred]
        supplement_buy_link = supplement_info['buy link'].iloc[pred]

        uploaded_image_url = url_for('static', filename=f'uploads/{filename}')
        return render_template(
            'submit.html',
            title=title,
            desc=description,
            prevent=prevent,
            image_url=image_url,
            pred=pred,
            sname=supplement_name,
            simage=supplement_image_url,
            buy_link=supplement_buy_link,
            uploaded_image=uploaded_image_url
        )
    return render_template('submit.html')

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template(
        'market.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy link'])
    )

if __name__ == '__main__':
    # On Render, PORT will be set by the platform
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
