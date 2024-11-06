import librosa
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score


def feature_extraction(file_path):
    x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=50).T, axis=0)
    return mfcc


# List to store features for each audio
features_list_c = []
features_list_tc = []
directory = '../audio_data/'

# Extract features from each audio file in the directory
for audio in os.listdir(directory):
    audio_path = os.path.join(directory, audio)
    if audio_path.endswith('.wav'):
        mfcc = feature_extraction(audio_path)
        if audio.startswith('c'):
            features_list_c.append([audio] + mfcc.tolist())
        elif audio.startswith('tc'):
            features_list_tc.append([audio] + mfcc.tolist())

# Convert list to numpy arrays
features_c = np.array(features_list_c)
features_tc = np.array(features_list_tc)

# Save numpy arrays to .npy files
np.save('audio_features_c.npy', features_c)
np.save('audio_features_tc.npy', features_tc)
print("Features saved to 'audio_features_c.npy' and 'audio_features_tc.npy'")

# Load data from .npy files
features_c = np.load('audio_features_c.npy')
features_tc = np.load('audio_features_tc.npy')

# Create DataFrames from numpy arrays
columns = ['filename'] + [f'mfcc_{i}' for i in range(50)]
df_c = pd.DataFrame(features_c, columns=columns)
df_tc = pd.DataFrame(features_tc, columns=columns)

# Add labels
df_c['label'] = 'cadel'
df_tc['label'] = 'tidak_cadel'

# Combine cadel and tidak cadel data
df = pd.concat([df_c, df_tc], ignore_index=True)

# Prepare data for training
X = df.iloc[:, 1:-1]  # Features (ignoring 'filename' and 'label' columns)
y = df['label']  # Labels

# Encode labels as integers
y = y.map({'cadel': 0, 'tidak_cadel': 1})

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build and train MLP model using scikit-learn
model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=50)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

# Save model
import joblib

joblib.dump(model, 'mlp_model.pkl')
print("Model saved to 'mlp_model.pkl'")
