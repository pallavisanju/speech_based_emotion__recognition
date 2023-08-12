import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define emotions and corresponding labels
emotions = {
    'angry': 0,
    'happy': 1,
    'sad': 2,
    'neutral': 3
}

# Function to extract features from audio file
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, duration=3)  # Load audio with a fixed duration
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    return np.mean(mfccs.T, axis=0)

# Load and preprocess dataset
X = []
y = []

data_dir = 'path_to_your_dataset_directory'
for emotion in emotions:
    emotion_dir = os.path.join(data_dir, emotion)
    for filename in os.listdir(emotion_dir):
        if filename.endswith('.wav'):
            audio_path = os.path.join(emotion_dir, filename)
            feature = extract_features(audio_path)
            X.append(feature)
            y.append(emotions[emotion])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Support Vector Machine
model = SVC()
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Example of real-time emotion prediction from microphone
import sounddevice as sd

def predict_emotion_live():
    print("Recording... (Press Ctrl+C to stop)")
    fs = 22050  # Sample rate
    seconds = 3  # Duration of recording

    audio_data = sd.rec(int(fs * seconds), samplerate=fs, channels=1)
    sd.wait()

    live_feature = extract_features(audio_data)
    live_emotion = model.predict([live_feature])[0]

    for emotion, label in emotions.items():
        if label == live_emotion:
            print(f"Predicted Emotion: {emotion}")

# Call the live emotion prediction function
predict_emotion_live()
