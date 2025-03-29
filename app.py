


from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all
# CORS(app)

# ✅ Model storage setup
MODEL_DIR = "./gpt2-business/trained"
MODEL_FILE = f"{MODEL_DIR}/model.safetensors"
MODEL_URL = "https://huggingface.co/LadlMe/gpt2-business/resolve/main/model.safetensors"


# ✅ Download model if not found
if not os.path.exists(MODEL_FILE):
    print("Downloading model...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_FILE, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print("Model downloaded!")

# ✅ Load the model
generator = pipeline("text-generation", model=MODEL_DIR, framework="pt")

def getResults(text):
    """Generates text using GPT-2 model"""
    output = generator(
        text,
        max_length=450,
        min_length=350,
        truncation=True,
        temperature=0.25,
        top_k=35,
        top_p=0.6,
        repetition_penalty=3.0,
        do_sample=True,
        num_return_sequences=2
    )

    return output

@app.route("/predict", methods=["POST"])
def predict():
    """Handles API requests for text generation"""
    data = request.get_json()
    
    if "text" not in data:
        return jsonify({"error": "Text is required"}), 400

    text = data["text"]
    results = {"text": getResults(text)[0]["generated_text"].replace(text, "", 1), "sender": "bot"}
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
