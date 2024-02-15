import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
import os
import librosa
from core import extract_features

# Create the dataset from the original audio files
# This is a one-time operation

# Load the audio files
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
genres_original_path = 'Data/genres_original'

audio_files = []
for genre in genres:
    genre_path = f'{genres_original_path}/{genre}'
    for filename in os.listdir(genre_path):
        
        try:
            # Load the audio file
            y, sr = librosa.load(f'{genre_path}/{filename}', duration=30)
        except Exception as e:
            print(f'Error processing {filename}: {str(e)}')
            continue
        
        audio_files.append({
            'filename': f'{genre_path}/{filename}',
            'length': librosa.get_duration(path=f'{genre_path}/{filename}'),
            'label': genre
        })
        print(f'Processed {filename}')

print(f'Processed {len(audio_files)} files')        

# Create the dataset
df = pd.DataFrame(audio_files)


# Extract features

print("Extracting features...")

# Apply feature extraction to the dataset and expand features into their own columns

feature_list = [extract_features(row) for row in df['filename']]
features_df = pd.DataFrame(feature_list)  # Convert list of numpy arrays into a DataFrame
df = pd.concat([df.reset_index(drop=True), features_df], axis=1)  # Concatenate along columns

# Print the dataset
print('Dataset:')
print(df.head())

#print('Features shape:', features_df.shape)

# Save the dataset

print('Saving dataset...')
df.to_csv('Data/features_30_sec.csv', index=False)

# Train the model

# Load dataset
data_path = 'Data/features_30_sec.csv'
df = pd.read_csv(data_path)

# Drop non-feature columns
X = df.drop(columns=['filename', 'length', 'label'])
y = df['label']

# Split dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('Training the model...')

# Initialize and train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


print('Model trained')

# Save the model and scaler

print('Saving model and scaler...')

dump(model, 'out/model.joblib')
dump(scaler, 'out/scaler.joblib')

print('Model and scaler saved')

print('Model evaluation')

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
