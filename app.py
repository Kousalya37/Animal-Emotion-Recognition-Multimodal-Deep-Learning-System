from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import os
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from PIL import Image
import open_clip
import openai

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")

fusion_model = tf.keras.models.load_model("fusion_model.h5")

device = "cuda" if torch.cuda.is_available() else "cpu"
wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model = clip_model.to(device)
clip_model.eval()

def extract_audio_embedding(audio_path, max_len=150):
    speech, sr = librosa.load(audio_path, sr=16000)
    inputs = wav2vec_processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec_model(**inputs).last_hidden_state.squeeze(0)
    seq = outputs[:max_len]
    if seq.shape[0] < max_len:
        padding = torch.zeros(max_len - seq.shape[0], seq.shape[1])
        seq = torch.cat([seq, padding], dim=0)
    return seq.cpu().numpy().reshape(1, max_len, seq.shape[1])

def extract_image_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(img_tensor)
    return embedding.cpu().numpy()

def simulate_biometric():
    hr = np.random.randint(60, 90, 1)
    temp = np.random.uniform(36, 38, 1)
    var = np.random.uniform(0.7, 1, 1)
    return np.array([hr[0], temp[0], var[0]]).reshape(1, -1)

def get_llm_explanation(emotion, prob_dist, user_text=None):
    prompt = f"""
    The detected emotion is: {emotion} with probabilities {prob_dist}.
    Generate a friendly, empathetic, and informative explanation or helpful advice based on this emotion.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
        )
        content = response.choices[0].message.content.strip()
        if not content or "I'm sorry" in content or "couldn't generate" in content:
            return f"The emotion seems to be {emotion}."
        return content
    except Exception as e:
        print("LLM API error:", e)
        return f"The emotion seems to be {emotion}."

@app.route('/predict_dataset', methods=["GET"])
def predict_dataset():
    try:
        audio_feats = np.load("audio_features_aligned.npy")
        image_feats = np.load("image_features_aligned.npy")
        bio_feats = np.load("biometric_features_aligned.npy")
    except Exception as e:
        return jsonify({"error": f"Feature files missing or corrupt: {e}"}), 500
    emotions = ['Happy', 'Sad', 'Angry', 'Fearful', 'Relaxed', 'Curious', 'Neutral']
    preds = fusion_model.predict([audio_feats, image_feats, bio_feats])
    predicted_classes = np.argmax(preds, axis=1)
    total = len(predicted_classes)
    histogram = {em: int((predicted_classes == i).sum()) for i, em in enumerate(emotions)}
    percentages = {em: round(histogram[em] * 100 / total, 2) for em in emotions}
    return jsonify({"histogram": histogram, "percentages": percentages})

@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files or "image" not in request.files:
        return jsonify({"error": "Audio and image required"}), 400
    audio_file = request.files["audio"]
    image_file = request.files["image"]

    audio_path = "temp_audio.wav"
    image_path = "temp_image.jpg"
    audio_file.save(audio_path)
    image_file.save(image_path)

    emotions = ['Happy', 'Sad', 'Angry', 'Fearful', 'Relaxed', 'Curious', 'Neutral']

    try:
        audio_feat = extract_audio_embedding(audio_path)
        image_feat = extract_image_embedding(image_path)
        biometric_feat = simulate_biometric()

        pred = fusion_model.predict([audio_feat, image_feat, biometric_feat])
        prob_dist = pred.flatten().tolist()
        idx = np.argmax(pred)
        confidence = float(np.max(pred))
        emotion = emotions[idx]

        llm_explanation = get_llm_explanation(emotion, prob_dist)

        return jsonify({
            "emotion": emotion,
            "confidence_percent": round(confidence * 100, 2),
            "llm_explanation": llm_explanation,
            "probability_distribution": {em: round(prob * 100, 2) for em, prob in zip(emotions, prob_dist)}
        })
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.exists(image_path):
            os.remove(image_path)

if __name__ == "__main__":
    app.run(debug=True)
