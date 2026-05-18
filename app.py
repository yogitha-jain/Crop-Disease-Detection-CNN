from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io

app = Flask(__name__)

# ── Class Names ──────────────────────────────────────────────
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

MODEL_PATH  = 'mobilenet_plantvillage.pth'
NUM_CLASSES = 38
IMG_SIZE    = 224
device      = torch.device("cpu")

# ── Load Model Once at Startup ────────────────────────────────
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

print("Loading model...")
model = load_model()
print("Model ready ✅")

# ── Preprocess ────────────────────────────────────────────────
def preprocess(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(img).unsqueeze(0)

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    image_bytes = file.read()
    tensor = preprocess(image_bytes).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs  = F.softmax(output, dim=1)

    top5_probs, top5_idx = torch.topk(probs, 5)

    top5 = []
    for i in range(5):
        cls  = CLASS_NAMES[top5_idx[0][i].item()]
        prob = round(top5_probs[0][i].item() * 100, 2)
        parts   = cls.split('___')
        plant   = parts[0].replace('_', ' ')
        disease = parts[1].replace('_', ' ').title() if len(parts) > 1 else ''
        top5.append({'plant': plant, 'disease': disease, 'confidence': prob})

    pred       = top5[0]
    is_healthy = 'healthy' in pred['disease'].lower()

    return jsonify({
        'plant':      pred['plant'],
        'disease':    pred['disease'],
        'confidence': pred['confidence'],
        'is_healthy': is_healthy,
        'top5':       top5
    })

if __name__ == '__main__':
    app.run(debug=True)