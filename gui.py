import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import os

# Paths to your model files
GENDER_MODEL_PATH = r'C:\Users\harsh\DESKTOP\Voice_Emotion_Detection\gender_detection_model.keras'
EMOTION_MODEL_PATH = r'C:\Users\harsh\DESKTOP\Voice_Emotion_Detection\emotion_detection_model.h5'
SCALER_GENDER_MEAN_PATH = r'C:\Users\harsh\DESKTOP\Voice_Emotion_Detection\gender_scaler_mean.npy'
SCALER_GENDER_STD_PATH = r'C:\Users\harsh\DESKTOP\Voice_Emotion_Detection\gender_scaler_var.npy'
SCALER_EMOTION_MEAN_PATH = r'C:\Users\harsh\DESKTOP\Voice_Emotion_Detection\emotion_scaler_mean.npy'
SCALER_EMOTION_STD_PATH = r'C:\Users\harsh\DESKTOP\Voice_Emotion_Detection\emotion_scaler_std.npy'

# Load models
try:
    model_gender = load_model(GENDER_MODEL_PATH)
    model_emotion = load_model(EMOTION_MODEL_PATH)

    # Load scalers
    scaler_gender_mean = np.load(SCALER_GENDER_MEAN_PATH)
    scaler_gender_std = np.load(SCALER_GENDER_STD_PATH)
    scaler_emotion_mean = np.load(SCALER_EMOTION_MEAN_PATH)
    scaler_emotion_std = np.load(SCALER_EMOTION_STD_PATH)
except Exception as e:
    print(f"Error loading models or scalers: {e}")

# Define GUI functions
def process_audio_file(file_path):
    try:
        # Load and preprocess the audio file
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs = np.expand_dims(mfccs, axis=-1)
        mfccs = np.expand_dims(mfccs, axis=0)

        # Predict gender
        features_gender = (mfccs - scaler_gender_mean) / scaler_gender_std
        gender_prediction = model_gender.predict(features_gender)
        if np.argmax(gender_prediction) != 1:  # Assuming 1 is the index for female
            messagebox.showerror("Gender Detection", "Upload female voice")
            return
        
        # Predict emotion
        features_emotion = (mfccs - scaler_emotion_mean) / scaler_emotion_std
        emotion_prediction = model_emotion.predict(features_emotion)
        emotion = np.argmax(emotion_prediction)
        emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise"]
        emotion_label = emotions[emotion]

        messagebox.showinfo("Emotion Detection", f"The emotion detected is: {emotion_label}")

    except Exception as e:
        messagebox.showerror("Error", f"Error processing file: {e}")

def record_audio():
    # Implement audio recording functionality
    pass

def upload_audio():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        process_audio_file(file_path)

def create_gui():
    root = tk.Tk()
    root.title("Voice Emotion Detection")

    upload_button = tk.Button(root, text="Upload Audio", command=upload_audio)
    upload_button.pack(pady=20)

    record_button = tk.Button(root, text="Record Audio", command=record_audio)
    record_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
