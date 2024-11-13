import librosa
import os
import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf


# from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.models import Sequential


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense


def feature_extraction(file_path):
    x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=50).T, axis=0)
    return mfcc


def train():
    # List untuk menyimpan fitur dari masing-masing audio
    features_list_c = []
    features_list_tc = []
    directory = '../audio_data/'

    # Iterasi melalui setiap file audio di direktori
    for audio in os.listdir(directory):
        audio_path = os.path.join(directory, audio)
        if audio_path.endswith('.wav'):
            mfcc = feature_extraction(audio_path)
            if audio.startswith('c'):
                features_list_c.append([audio] + mfcc.tolist())
            elif audio.startswith('tc'):
                features_list_tc.append([audio] + mfcc.tolist())

    # Konversi list ke numpy array
    features_c = np.array(features_list_c)
    features_tc = np.array(features_list_tc)

    # Simpan numpy array ke file .npy
    np.save('audio_features_c.npy', features_c)
    np.save('audio_features_tc.npy', features_tc)

    print("Features saved to 'audio_features_c.npy' and 'audio_features_tc.npy'")

    # Load data dari file .npy
    features_c = np.load('audio_features_c.npy')
    features_tc = np.load('audio_features_tc.npy')

    # Membuat DataFrame dari numpy array
    columns = ['filename'] + [f'mfcc_{i}' for i in range(50)]
    df_c = pd.DataFrame(features_c, columns=columns)
    df_tc = pd.DataFrame(features_tc, columns=columns)

    # Tambahkan label
    df_c['label'] = 'cadel'
    df_tc['label'] = 'tidak_cadel'

    # Gabungkan data cadel dan tidak cadel
    df = pd.concat([df_c, df_tc], ignore_index=True)

    # Siapkan data untuk pelatihan
    X = df.iloc[:, 1:-1]  # Fitur (mengabaikan kolom 'filename' dan 'label')
    y = df['label']  # Label

    # Encode labels menjadi integers
    y = y.map({'cadel': 0, 'tidak_cadel': 1})

    # Normalisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build model
    model = Sequential()
    model.add(Dense(100, input_dim=50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    # Change epochs to increase accuracy
    model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy}")

    # Save model
    model.save('mlp_model.keras')
    print("Model saved to 'mlp_model.keras'")

    result = ""

    if accuracy != 0:
        result = f"Training Successfully.\nAccuracy: {accuracy}"

    return result
