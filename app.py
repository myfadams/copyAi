from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os

app = Flask(__name__)

# ✅ Properly configure CORS
CORS(app, resources={r"/predict": {"origins": "*"}}, supports_credentials=True)

# ✅ Load the model directly from Hugging Face
MODEL_NAME = "LadlMe/gpt2-business"
generator = pipeline("text-generation", model=MODEL_NAME, framework="pt")

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

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """Handles API requests for text generation"""
    
    # ✅ Handle preflight (OPTIONS) requests
    if request.method == "OPTIONS":
        response = jsonify({"message": "CORS preflight successful"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return response, 200
    
    data = request.get_json()
    
    if "text" not in data:
        return jsonify({"error": "Text is required"}), 400

    text = data["text"]
    results = {"text": getResults(text)[0]["generated_text"].replace(text, "", 1), "sender": "bot"}

    # ✅ Add CORS headers to the response
    response = jsonify(results)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
