
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import pipeline

# # Load the fine-tuned model
# generator = pipeline("text-generation", model="./gpt2-business/trained", framework="pt")

# def getResults(text):
#         # Generate a business-related text sample
#     output = generator(
#         text,
#         max_length=450,  # More room to finish thoughts
#         min_length=350,
#         truncation=True,
#         temperature=0.25,  # Makes output more structured
#         top_k=35,  # Keeps responses diverse but relevant
#         top_p=0.6,  # Restricts randomness
#         repetition_penalty=3.0,  # Reduces repetitive phrases
#         do_sample=True,
#         num_return_sequences=2  # Generates two responses to pick from
#     )

#     # output = generator(
#     #     text,
#     #     max_length=200,  # More room for complete thoughts
#     #     truncation=True,
#     #     temperature=0.25,  # Even more controlled
#     #     top_k=30,  # Slightly higher for better word choice
#     #     top_p=0.6,  # Further restricts randomness
#     #     repetition_penalty=3.0,  # Stronger penalty to avoid loops
#     #     do_sample=True,
#     #     num_return_sequences=2  # Get two outputs and choose the best
#     # )

#     print(jsonify(output))
#     return output
# # Flask app
# app = Flask(__name__)

# CORS(app)


# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()
    
#     if "text" not in data:
#         return jsonify({"error": "Text is required"}), 400

#     text = data["text"]  # Clean the text
#     results={ "text": getResults(text)[0]["generated_text"].replace(text, "", 1), "sender": "bot" }
#     return jsonify(results)

# if __name__ == "__main__":
#     app.run(debug=True)




from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os
import requests

app = Flask(__name__)
CORS(app)

# ✅ Model storage setup
MODEL_DIR = "./gpt2-business/trained"
MODEL_FILE = f"{MODEL_DIR}/model.safetensors"
MODEL_URL = "https://huggingface.co/path-to-your-model/model.safetensors"  # Change this to your model link

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
