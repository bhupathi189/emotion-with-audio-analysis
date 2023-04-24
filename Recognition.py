import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
import os
import sys
import pickle
import librosa
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from IPython.display import Audio
from tensorflow import keras
from flask import Flask, render_template, request
import boto3

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


app = Flask(__name__)

obj = boto3.client(
    "s3",
    aws_access_key_id="AKIA23X2YQJO5QNCW6LY",
    aws_secret_access_key="4bectTs5hG9IAaH2J9ZjtDkbhQch0Y3EwBf9jGoN"
    # ,    aws_session_token=SESSION_TOKEN
)


def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result


def get_features(path, sample_rate):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3))  # stacking vertically

    return result


def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data


def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(y=data, rate=rate)


def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)


Y = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()

# src = "03-01-02-01-01-02-05.wav"


@app.route("/predict", methods=["POST"])
def predict_placement():
    file_name = request.headers["name"]

    # Downloading a csv file
    # from S3 bucket to local folder
    obj.download_file(
        Filename="./AudioFiles/" + file_name,
        Bucket="sagemaker-audio-files",
        Key=file_name,
    )
    src = "./AudioFiles/" + file_name
    data, sample_rate = librosa.load(src)
    Feature_list = get_features(src, sample_rate)

    scaler = pickle.load(open("scaler.pkl", "rb"))
    loaded_model = load_model("savedmodel.h5")
    # loaded_model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

    print("Not scaled  --> ", Feature_list)
    scaled_features = scaler.transform(Feature_list)
    print("Scaled  --> ", scaled_features)

    scaled_features = pd.DataFrame(scaled_features)
    Features = np.expand_dims(scaled_features, axis=2)
    print(Features.shape)

    predicted_feature = loaded_model.predict(Features)
    print(predicted_feature)

    y_pred = encoder.inverse_transform(predicted_feature)

    print(y_pred.flatten())

    Feature_list = []
    return y_pred.flatten()[0]


# predicted_feature = loaded_model.predict(Feature_list[1])
# print(predicted_feature)

# predicted_feature = loaded_model.predict(Feature_list[2])
# print(predicted_feature)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)