from flask import Flask, render_template, request
import pickle
import numpy as np
import soundfile
import librosa
import boto3

model = pickle.load(open("sklearn_model.pkl", "rb"))
app = Flask(__name__)

obj = boto3.client(
    "s3",
    aws_access_key_id="AKIA23X2YQJO5QNCW6LY",
    aws_secret_access_key=""
    # ,    aws_session_token=SESSION_TOKEN
)


def extract_feature(file_name, mfcc=True, chroma=True, mel=True, zero_crossing=True):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])
        if mfcc:
            mfccs = np.mean(
                librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0
            )
            result = np.hstack((result, mfccs))
        if chroma:
            stft = np.abs(librosa.stft(X))
            chroma = np.mean(
                librosa.feature.chroma_stft(S=stft, sr=sample_rate, n_chroma=24).T,
                axis=0,
            )
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        if zero_crossing:
            zc = sum(librosa.zero_crossings(X, pad=False))
            result = np.hstack((result, zc))
    return result


@app.route("/")
def index():
    return render_template("index.html")


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

    # prediction
    result = model.predict(np.array(extract_feature("./AudioFiles/" + file_name)).reshape(1, -1))

    # if result[0] == 'sad':
    #     result = '😔'
    # else:
    #     result = 'not placed'

    # return render_template('index.html',result=result)
    return result[0]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
