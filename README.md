Welcome to the English Accent Detector — a simple, intuitive web app that can analyze the accent of English speakers from public videos, such as YouTube clips. Just paste a video link, and the app does the rest: it extracts the audio, processes the speech, and tells you which English accent it likely is!

Try it live here: https://accent.streamlit.app/

How It Works
Our app combines powerful modern tools from the worlds of machine learning, audio processing, and web development to deliver fast, accurate accent detection:

Key Technologies:
Streamlit:
A super-friendly Python framework that lets us build and deploy interactive web apps quickly. It handles the UI, user input, and live updates seamlessly.

yt-dlp:
A versatile downloader that fetches audio and video from public URLs like YouTube. It grabs the best quality audio from your provided link.

FFmpeg:
The industry-standard multimedia toolkit used to convert and process audio and video files. We rely on FFmpeg to extract audio from videos and convert it into a clean WAV format at 16kHz — perfect for speech models.

Librosa:
A Python library for audio analysis. It loads the audio in the correct format and sampling rate to prepare for feature extraction.

Facebook Wav2Vec2 Model:
This is a state-of-the-art speech representation model from Facebook AI. It converts raw audio into meaningful numerical features (embeddings) that capture speech patterns relevant to accent.

Logistic Regression Classifier:
A lightweight machine learning model trained on labeled accent data. It takes Wav2Vec2 features and predicts the speaker's English accent with probabilities for each class.

Joblib:
Used for loading the trained model and label encoder efficiently.

Features
Video URL input: Accepts any public video link to analyze the spoken English accent.

Automatic audio extraction: Downloads and converts video audio seamlessly behind the scenes.

Accurate accent prediction: Classifies accents like American, British, Australian, Indian, and more.

Confidence scores: Shows how confident the model is and provides a detailed breakdown.

Interactive UI: Easy-to-use interface with real-time feedback and results.

Why This Matters
Understanding accents can be valuable in language learning, hiring, or even accessibility contexts. This app provides an accessible way to explore the rich variety of English accents worldwide — all from any online video.

Try It Out
Open https://accent.streamlit.app/, paste your video URL, and hit Analyze to see which English accent is detected!

Technical Notes
This app requires ffmpeg to be installed on the server or environment where it runs. On Streamlit Cloud, ffmpeg is pre-installed.

The model currently supports a set of common English accents based on the training data used.

Processing time depends on video length and server load but usually takes just a few seconds.
