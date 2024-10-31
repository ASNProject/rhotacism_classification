# import librosa
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
#
# # Load model
# model = tf.keras.models.load_model('mlp_model.keras')
#
# # Load scaler (assuming you saved and loaded it correctly)
# scaler = StandardScaler()
#
# # Load or define X_train and y_train from your training data
# # For example, if you have saved them in numpy arrays:
# features_c = np.load('audio_features_c.npy')
# features_tc = np.load('audio_features_tc.npy')
#
# # Create DataFrames from numpy arrays
# columns = ['filename'] + [f'mfcc_{i}' for i in range(50)]
# df_c = pd.DataFrame(features_c, columns=columns)
# df_tc = pd.DataFrame(features_tc, columns=columns)
#
# # Add labels to DataFrames
# df_c['label'] = 'cadel'
# df_tc['label'] = 'tidak_cadel'
#
# # Concatenate dataframes
# df = pd.concat([df_c, df_tc], ignore_index=True)
#
# # Prepare data for training
# X = df.iloc[:, 1:-1].values  # Features (excluding 'filename' and 'label' columns)
# y = df['label'].map({'cadel': 0, 'tidak_cadel': 1}).values  # Labels
#
# # Fit scaler on training data
# scaler.fit(X)
#
#
# # Function for feature extraction from new audio data
# def feature_extraction(file_path):
#     x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
#     mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=50).T, axis=0)
#     return mfcc
#
#
# # Example new audio file (ensure its format matches the training data)
# new_audio_file = '../audio_data/tc1.wav'
#
# # Extract features from new audio data
# new_features = feature_extraction(new_audio_file)
#
# # Reshape new features to match the expected input shape of the model
# new_features_reshaped = new_features.reshape(1, -1)  # Assuming you have a single sample
#
# # Normalize new features using the previously fitted scaler
# new_features_scaled = scaler.transform(new_features_reshaped)
#
# # Perform prediction using the model
# prediction = model.predict(new_features_scaled)
#
# # Output prediction result
# if prediction[0] >= 0.5:
#     print("Audio tidak cadel")
# else:
#     print("Audio cadel")
#
#
# # Visualize the MFCC features
# plt.figure(figsize=(10, 4))
# plt.imshow(new_features.reshape(1, -1), interpolation='nearest', cmap='viridis', aspect='auto')
# plt.title('MFCC Features of New Audio Data')
# plt.xlabel('MFCC Coefficients')
# plt.ylabel('Feature')
# plt.colorbar()
# plt.tight_layout()
# plt.show()

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Function for feature extraction from new audio data
def feature_extraction(file_path):
    x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=50).T, axis=0)
    return mfcc


def prediction_mlp():
    # Load model
    model = tf.keras.models.load_model('mlp_model.keras')

    # Load scaler
    scaler = StandardScaler()

    # Load or define X_train and y_train from your training data
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

    # Example new audio file
    new_audio_file = 'output.wav'

    # Extract features from new audio data
    new_features = feature_extraction(new_audio_file)

    # Print extracted features
    print("Extracted MFCC features from new audio data:")
    for i, value in enumerate(new_features, start=1):
        print(f"MFCC_{i}: {value}")

    # Reshape new features
    new_features_reshaped = new_features.reshape(1, -1)

    # Normalize new features
    new_features_scaled = scaler.transform(new_features_reshaped)

    # Perform prediction
    prediction = model.predict(new_features_scaled)
    probability_normal = prediction[0][0]  # Probability of the positive class (tidak_cadel)
    probability_cadel = 1 - probability_normal  # Probability of the negative class (cadel)

    # Output prediction results and percentages
    print(f"Probability of 'tidak cadel': {probability_normal * 100:.2f}%")
    print(f"Probability of 'cadel': {probability_cadel * 100:.2f}%")

    # Determine status based on the threshold
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
    # plt.show()
    return status, probability_cadel, probability_normal

