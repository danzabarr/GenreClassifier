import librosa
from joblib import load
import numpy as np
import pandas as pd


from core import extract_features


# Load your pre-trained model (adjust this with your actual model loading code)
model = load('out/model.joblib')

# Load the scaler
scaler  = load('out/scaler.joblib') 

# what is the scaler used for?
# The scaler is used to scale the features before they are fed into the model.

audio_path = "test/Howlin' Wolf - Smokestack Lightnin'.wav"
#audio_path = 'Data/genres_original/blues/blues.00001.wav'
print(f'Loading {audio_path}...')

print(f'Extracting features')
features = extract_features(audio_path).reshape(1, -1);

# Scale the features
print(f'Scaling features')
features = scaler.transform(features)

print(f'Predicting genre...')
# Make a prediction 
#prediction = model.predict(features)
#print(f'Prediction: {prediction[0]}')

# get the scores for each genre
scores = model.predict_proba(features)[0]
genre_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
zipped = zip(genre_names, scores)
sorted = sorted(zipped, key=lambda x: x[1], reverse=True)

for genre, score in sorted:
    print(f'{genre: <10} {score:.2f}')
