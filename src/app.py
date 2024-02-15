from flask import Flask, request, jsonify
from pytube import YouTube
import os
import librosa
import numpy as np
from joblib import load
from pydub import AudioSegment
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from core import extract_features
import os
# Rest of your imports and model loading code

app = Flask(__name__, static_folder='static', template_folder='../templates')

@app.route('/')
def index():
    return render_template('index.html')  # Ensure 'index.html' is in the 'templates' folder of your project

CORS(app)

# print the templates folder for the app
print(app.template_folder)


# Load your pre-trained model (adjust this with your actual model loading code)
model = load('out/model.joblib')

@app.route('/predict_genre', methods=['POST'])
def predict_genre():
    data = request.json
    youtube_url = data.get('youtube_url')
    
    if not youtube_url:
        return jsonify({'error': 'Missing YouTube URL'}), 400

    try:
        # Download audio using PyTube
        yt = YouTube(youtube_url)
        
        print(f'Downloading audio from {youtube_url}')
        stream = yt.streams.filter(only_audio=True).first()
        download_path = stream.download()
        
        print(f'Converting to WAV format: {audio_path}')
        audio_path = convert_to_wav(download_path)


        # Extract features and predict genre (adjust these functions according to your project)
        print(f'Extracting features')
        features = extract_features(audio_path)
        
        # Scale the features
        print(f'Scaling features')
        scaler  = load('out/scaler.joblib')
        features = scaler.transform(features)

        # Make a prediction
        genre = model.predict(features)[0]        
        print(f'Predicted genre: {genre}')

        # get the scores for each genre
        scores = model.predict_proba(features)[0]
        print(f'Scores: {scores}')

        # Clean up: remove downloaded files after processing
        os.remove(download_path)
        os.remove(audio_path)

        return jsonify({'genre': genre})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def convert_to_wav(audio_file_path):
    # Load the audio file
    audio = AudioSegment.from_file(audio_file_path)

    # Ensure the audio is in WAV format
    if audio_file_path.endswith('.wav'):
        return audio_file_path  # Already in WAV format

    # Define the output file path (replace the extension with '.wav')
    wav_file_path = os.path.splitext(audio_file_path)[0] + '.wav'

    # Export the audio in WAV format
    audio.export(wav_file_path, format='wav')

    return wav_file_path


if __name__ == '__main__':
    app.run(debug=True)
