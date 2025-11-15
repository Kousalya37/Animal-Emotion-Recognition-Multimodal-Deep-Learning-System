import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
import os

wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
def get_audio_embedding(audio_path, max_len=150):
    speech, sr = librosa.load(audio_path, sr=16000)
    inputs = wav2vec_processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec_model(**inputs).last_hidden_state.squeeze(0)
    seq = outputs[:max_len]
    if seq.shape[0] < max_len:
        padding = torch.zeros(max_len - seq.shape[0], seq.shape[1])
        seq = torch.cat([seq, padding], dim=0)
    return seq.numpy()

def extract_audio_features_recursive(folder_path, max_len=150):
    features = []
    file_list = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(('.wav', '.mp3')):
                full_path = os.path.join(root, f)
                print(f"Found audio file: {full_path}")
                feat = get_audio_embedding(full_path, max_len)
                features.append(feat)
                file_list.append(full_path)
    if features:
        features = np.array(features)
        np.save("audio_sequence_features.npy", features)
        with open("audio_files.txt", "w") as f:
            for file in file_list:
                f.write(file + "\n")
        print(f"Extracted {len(features)} audio embeddings and saved to audio_sequence_features.npy")
    else:
        print(f"No audio files found in {folder_path}")
    return features

if __name__ == "__main__":
    audio_folder = "audio_data" 
    extract_audio_features_recursive(audio_folder)
