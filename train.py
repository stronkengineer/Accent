import torch
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Extract features using wav2vec2 from waveform numpy array
def extract_features(waveform_array, processor, model):
    inputs = processor(waveform_array, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

# Load the Common Accent dataset from Hugging Face, get waveform arrays and labels
def load_accent_dataset(max_samples=100):
    dataset = load_dataset("DTU54DL/common-accent")
    train_data = dataset["train"].select(range(max_samples))
    # Each sample's "audio" is a dict with "array" (numpy waveform) and "sampling_rate"
    data = [(sample["audio"]["array"], sample["accent"]) for sample in train_data]
    print(f"Loaded {len(data)} labeled samples.")
    print(f"Accents found: {set(label for _, label in data)}")
    return data

def main():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model.eval()

    data = load_accent_dataset(max_samples=100)

    X, y = [], []
    for waveform_array, label in data:
        features = extract_features(waveform_array, processor, model)
        X.append(features)
        y.append(label)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    # Specify labels explicitly to avoid mismatch errors
    print(classification_report(y_test, y_pred, target_names=le.classes_, labels=range(len(le.classes_))))

    joblib.dump((clf, le), "accent_classifier.pkl")
    print("Training complete and model saved as accent_classifier.pkl")

if __name__ == "__main__":
    main()
