from flask import Flask, render_template, request
import boto3
# import pickle
import numpy as np
import soundfile
import librosa

from keras.models import load_model

# model = pickle.load(open("sklearn_model.pkl", "rb"))
model = load_model('savedmodel.h5')
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


# new code


def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)
    
    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2)) # stacking vertically
    
    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3)) # stacking vertically
    
    return result

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


Y = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

src = '/Users/divya/Desktop/neutral.wav'

data, sample_rate = librosa.load(src)
Feature_list = get_features(src)

# scaler  = pickle.load(open('scaler.pkl','rb'))
loaded_model = load_model("savedmodel.h5")
# loaded_model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

print("Not scaled  --> ",Feature_list)
scaled_features = scaler.transform(Feature_list)
print("Scaled  --> ",scaled_features)

scaled_features = pd.DataFrame(scaled_features)
Features = np.expand_dims(scaled_features, axis=2)
print(Features.shape)

predicted_feature = loaded_model.predict(Features)
print(predicted_feature)

y_pred = encoder.inverse_transform(predicted_feature)

print(y_pred.flatten())

Feature_list = []

# @app.route("/")
# def index():
#     return render_template("index.html")


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
