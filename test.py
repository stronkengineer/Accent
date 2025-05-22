import streamlit as st
import torch
import torchaudio
import subprocess
import os
import uuid
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import joblib
import numpy as np
from yt_dlp import YoutubeDL
import imageio_ffmpeg

# Streamlit setup
st.set_page_config(page_title="Accent Classifier", layout="centered")
st.title("🎙️ English Accent Detector")

# Load Wav2Vec2
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

# Load classifier and encoder
@st.cache_resource
def load_classifier():
    clf, encoder = joblib.load("accent_classifier.pkl")
    return clf, encoder

clf, encoder = load_classifier()

# Download and convert video/audio
def download_audio_from_video(url, output_dir="temp"):
    os.makedirs(output_dir, exist_ok=True)
    temp_id = str(uuid.uuid4())
    video_path = os.path.join(output_dir, f"{temp_id}.mp4")
    audio_path = os.path.join(output_dir, f"{temp_id}.wav")
    
    # Download video using yt-dlp
    ydl_opts = {
        'outtmpl': video_path,
        'format': 'bestaudio/best',
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not os.path.exists(video_path):
        raise RuntimeError(f"Video file not found after download: {video_path}")

    # Use ffmpeg from imageio-ffmpeg package
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    result = subprocess.run([
        ffmpeg_path, "-y", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    os.remove(video_path)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr.strip()}")

    return audio_path

# Extract Wav2Vec2 features
def extract_features(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path, backend="soundfile")
    if sample_rate != 16000:
        resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample(waveform)
    inputs = processor(waveform[0], sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

# Predict accent
def classify_accent(features, clf, encoder):
    probs = clf.predict_proba([features])[0]
    top_idx = np.argmax(probs)
    label = encoder.inverse_transform([top_idx])[0]
    return label, probs[top_idx], probs

# UI
video_url = st.text_input("Enter a public video URL (e.g., YouTube):")

if st.button("Analyze") and video_url:
    with st.spinner("Downloading and processing audio..."):
        try:
            audio_path = download_audio_from_video(video_url)
            features = extract_features(audio_path)
            accent, confidence, all_probs = classify_accent(features, clf, encoder)
            os.remove(audio_path)
        except Exception as e:
            st.error(f"Failed to process video: {e}")
            st.stop()

    st.success(f"Accent Detected: **{accent.title()} English**")
    st.metric("Confidence", f"{confidence * 100:.2f}%")

    with st.expander("Detailed Confidence Scores"):
        decoded_labels = encoder.inverse_transform(np.arange(len(all_probs)))
        st.bar_chart({label: prob for label, prob in zip(decoded_labels, all_probs)})
