import librosa

def extract_features(audio_path, reshape=False):
    print(f'Extracting features from {audio_path}')
    y, sr = librosa.load(audio_path, duration=30)
    # Extract features, for example, MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    # Flatten the features array
    mfcc_flat = mfcc.flatten()
    # Return the features as a numpy array
    
    return mfcc_flat;
    
    #return mfcc_flat.reshape(1, -1)  # Reshape for a single sample
