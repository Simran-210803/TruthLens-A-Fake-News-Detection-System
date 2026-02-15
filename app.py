# from flask import Flask, request, render_template, jsonify
# from werkzeug.utils import secure_filename
# import os
# import requests
# from PIL import Image
# import torch
# import torch.nn as nn
# from torchvision import transforms

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'

# # === Google-based Fake News Detection ===
# def detect_fake_news(query): 
#     CREDIBLE_DOMAINS = ['bbc.com', 'cnn.com', 'reuters.com', 'theguardian.com', 'nytimes.com', '.gov', '.edu']
#     FAKE_DOMAINS = ['fakenews.com', 'theonion.com', 'infowars.com', 'clickhole.com', 'babylonbee.com', 'dailybuzzlive.com']
#     SUSPICIOUS_KEYWORDS = ['shocking', 'you won’t believe', 'miracle', 'cure', 'conspiracy', 'aliens', 'secret', 'banned', 'truth revealed', 'deep state', 'plandemic', 'hoax', 'fake', 'bizarre']

#     try:
#         search_url = (
#             f"https://www.googleapis.com/customsearch/v1?q={query}"
#             f"&key=YourGoogleAPIKeyHere"
#             f"&cx=your_cx_id"
#         )
#         response = requests.get(search_url)
#         data = response.json()
#         items = data.get('items', [])
#         if not items:
#             return "Likely Fake", [], 10

#         credible_hits = 0
#         fake_hits = 0
#         links = []

#         for item in items[:5]:
#             title = item.get('title', 'No Title')
#             link = item.get('link', '')
#             links.append((title, link))

#             if any(domain in link for domain in CREDIBLE_DOMAINS):
#                 credible_hits += 1
#             if any(domain in link for domain in FAKE_DOMAINS):
#                 fake_hits += 1

#         keyword_hits = sum(1 for word in SUSPICIOUS_KEYWORDS if word.lower() in query.lower())

#         if fake_hits >= 2 or keyword_hits >= 3:
#             return "Likely Fake", links, 15
#         elif fake_hits == 1 or keyword_hits >= 2:
#             return "Possibly Fake", links, 35
#         elif credible_hits >= 3:
#             return "Likely Real", links, 90
#         elif credible_hits >= 1:
#             return "Possibly Real", links, 60
#         else:
#             return "Likely Fake", links, 30
#     except Exception as e:
#         print("Error retrieving data:", e)
#         return "Error retrieving data", [], 0

# # === CNN Model Class for Morphing Detection ===
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(32 * 64 * 64, 128),
#             nn.ReLU(),
#             nn.Linear(128, 2)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x

# # === Load Trained PyTorch Model ===
# morph_model = SimpleCNN()
# morph_model.load_state_dict(torch.load('morph_model.pth', map_location=torch.device('cpu')))
# morph_model.eval()

# # === Real Image Morphing Detection Function ===
# def detect_image_morphing(image_path):
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor()
#     ])
#     img = Image.open(image_path).convert('RGB')
#     img_tensor = transform(img).unsqueeze(0)
#     with torch.no_grad():
#         output = morph_model(img_tensor)
#         probs = torch.softmax(output, dim=1)
#         pred = torch.argmax(probs).item()
#         label = "Morphed" if pred == 0 else "Original"
#         confidence = round(float(probs[0][pred]) * 100, 2)
#         return label, confidence

# # === Routes ===
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     user_input = request.form.get('user_input', '').strip()
#     if not user_input:
#         return jsonify({'error': 'No input provided'}), 400

#     result, links, accuracy = detect_fake_news(user_input)
#     if result == "Error retrieving data":
#         return jsonify({'error': 'Could not fetch prediction from Google API'}), 500

#     return jsonify({'prediction': result, 'links': links, 'accuracy': accuracy})

# @app.route('/predict_image', methods=['POST'])
# def predict_image():
#     if 'image_file' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400

#     file = request.files['image_file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     filename = secure_filename(file.filename)
#     os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(filepath)

#     result, accuracy = detect_image_morphing(filepath)
#     return jsonify({
#         'prediction': result,
#         'accuracy': accuracy,
#         'image_path': '/' + filepath.replace('\\', '/')
#     })

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import requests
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms  
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
CX = os.getenv("GOOGLE_CX")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# === Google-based Fake News Detection with Suspicious Keywords and Likely/Possibly Labels ===
def detect_fake_news(query):
    CREDIBLE_DOMAINS = [
        'bbc.com', 'cnn.com', 'reuters.com', 'nytimes.com', 'theguardian.com',
        'indiatoday.in', 'ndtv.com', 'thehindu.com', 'hindustantimes.com',
        'timesofindia.indiatimes.com', '.gov', '.edu'
    ]

    FAKE_DOMAINS = [
        'fakenews.com', 'theonion.com', 'infowars.com',
        'clickhole.com', 'babylonbee.com', 'dailybuzzlive.com'
    ]

    SUSPICIOUS_KEYWORDS = [
        'shocking', 'you won’t believe', 'miracle', 'cure', 'conspiracy',
        'aliens', 'secret', 'banned', 'truth revealed', 'deep state',
        'plandemic', 'hoax', 'fake', 'bizarre', 'exposed', 'urgent',
        'viral', 'must watch', 'alert', 'danger', 'banned video'
    ]

    try:
        search_url = (
            f"https://www.googleapis.com/customsearch/v1?q={query}"
            f"&key={API_KEY}"
            f"&cx={CX}"
        )

        response = requests.get(search_url, timeout=5)
        data = response.json()
        items = data.get('items', [])

        credible_hits = 0
        fake_hits = 0
        links = []

        for item in items[:5]:
            title = item.get('title', 'No Title')
            link = item.get('link', '')
            links.append((title, link))

            if any(domain in link for domain in CREDIBLE_DOMAINS):
                credible_hits += 1
            if any(domain in link for domain in FAKE_DOMAINS):
                fake_hits += 1

        keyword_hits = sum(
            1 for word in SUSPICIOUS_KEYWORDS
            if word.lower() in query.lower()
        )

        if fake_hits >= 2 or keyword_hits >= 4:
            return "Likely Fake", links, 15
        elif fake_hits == 1 or keyword_hits == 3:
            return "Possibly Fake", links, 35
        elif credible_hits >= 3 and keyword_hits <= 1:
            return "Likely Real", links, 90
        elif credible_hits >= 1 and keyword_hits <= 2:
            return "Possibly Real", links, 60
        else:
            return "Likely Fake", links, 30

    except Exception as e:
        print("Error retrieving data:", e)
        return "Error retrieving data", [], 0


# === CNN Model Class for Morphing Detection ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# === Load Trained PyTorch Model ===
morph_model = SimpleCNN()
morph_model.load_state_dict(
    torch.load('morph_model.pth', map_location=torch.device('cpu'))
)
morph_model.eval()


# === Real Image Morphing Detection Function ===
def detect_image_morphing(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = morph_model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs).item()

        label = "Morphed" if pred == 0 else "Original"
        confidence = round(float(probs[0][pred]) * 100, 2)

        return label, confidence


# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form.get('user_input', '').strip()
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    result, links, accuracy = detect_fake_news(user_input)

    if result == "Error retrieving data":
        return jsonify({'error': 'Could not fetch prediction from Google API'}), 500

    return jsonify({
        'prediction': result,
        'links': links,
        'accuracy': accuracy
    })


@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'image_file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image_file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result, accuracy = detect_image_morphing(filepath)

    return jsonify({
        'prediction': result,
        'accuracy': accuracy,
        'image_path': '/' + filepath.replace('\\', '/')
    })


if __name__ == '__main__':
    app.run(debug=True)
