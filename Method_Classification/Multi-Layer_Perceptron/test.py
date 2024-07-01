import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load model
model = tf.keras.models.load_model('mlp_model.keras')

# Load scaler (assuming you saved and loaded it correctly)
scaler = StandardScaler()

# Load or define X_train and y_train from your training data
# For example, if you have saved them in numpy arrays:
features_c = np.load('audio_features_c.npy')
features_tc = np.load('audio_features_tc.npy')

# Create DataFrames from numpy arrays
columns = ['filename'] + [f'mfcc_{i}' for i in range(50)]
df_c = pd.DataFrame(features_c, columns=columns)
df_tc = pd.DataFrame(features_tc, columns=columns)

# Add labels to DataFrames
df_c['label'] = 'cadel'
df_tc['label'] = 'tidak_cadel'

# Concatenate dataframes
df = pd.concat([df_c, df_tc], ignore_index=True)

# Prepare data for training
X = df.iloc[:, 1:-1].values  # Features (excluding 'filename' and 'label' columns)
y = df['label'].map({'cadel': 0, 'tidak_cadel': 1}).values  # Labels

# Fit scaler on training data
scaler.fit(X)


# Function for feature extraction from new audio data
def feature_extraction(file_path):
    x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=50).T, axis=0)
    return mfcc


# Example new audio file (ensure its format matches the training data)
new_audio_file = '../audio_data/tc1.wav'

# Extract features from new audio data
new_features = feature_extraction(new_audio_file)

# Reshape new features to match the expected input shape of the model
new_features_reshaped = new_features.reshape(1, -1)  # Assuming you have a single sample

# Normalize new features using the previously fitted scaler
new_features_scaled = scaler.transform(new_features_reshaped)

# Perform prediction using the model
prediction = model.predict(new_features_scaled)

# Output prediction result
if prediction[0] >= 0.5:
    print("Audio tidak cadel")
else:
    print("Audio cadel")
