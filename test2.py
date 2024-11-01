import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import joblib

# Feature extraction function
def feature_extraction(file_path):
    x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=50).T, axis=0)
    return mfcc

# Load data and prepare DataFrame as before
features_c = np.load('audio_features_c.npy')
features_tc = np.load('audio_features_tc.npy')

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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, activation='relu', solver='adam')
mlp.fit(X_scaled, y)

# Save the model and scaler
joblib.dump(mlp, 'mlp_model_sklearn.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("Model and scaler saved.")

# Prediction function
def prediction_mlp():
    # Load the model and scaler
    mlp = joblib.load('mlp_model_sklearn.joblib')
    scaler = joblib.load('scaler.joblib')

    # Example new audio file
    new_audio_file = 'output.wav'

    # Extract features from new audio data
    new_features = feature_extraction(new_audio_file)
    print("Extracted MFCC features from new audio data:")
    for i, value in enumerate(new_features, start=1):
        print(f"MFCC_{i}: {value}")

    # Reshape and normalize new features
    new_features_reshaped = new_features.reshape(1, -1)
    new_features_scaled = scaler.transform(new_features_reshaped)

    # Perform prediction
    probability = mlp.predict_proba(new_features_scaled)
    probability_cadel = probability[0][0]
    probability_normal = probability[0][1]

    # Output prediction results and percentages
    print(f"Probability of 'tidak cadel': {probability_normal * 100:.2f}%")
    print(f"Probability of 'cadel': {probability_cadel * 100:.2f}%")

    if probability_normal >= 0.5:
        print(f"Audio tidak cadel (Probability: {probability_normal * 100:.2f}%)")
        status = "Normal"
    else:
        print(f"Audio cadel (Probability: {probability_cadel * 100:.2f}%)")
        status = "Cadel"

    # Visualize the MFCC features
    plt.figure(figsize=(12, 6))
    plt.plot(new_features, label='MFCC')
    plt.title('Average Amplitude of MFCC Features of New Audio Data')
    plt.xlabel('MFCC Coefficients')
    plt.ylabel('Average Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.savefig('mfcc_new_audio_average_amplitude_plot.png')
    return status, probability_cadel, probability_normal

# Example usage
status, prob_cadel, prob_normal = prediction_mlp()
print(f"Status: {status}, Probability Cadel: {prob_cadel:.2f}, Probability Normal: {prob_normal:.2f}")
