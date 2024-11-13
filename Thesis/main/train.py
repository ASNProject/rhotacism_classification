import librosa
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


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
    print(X_scaled)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build model
    mlp = MLPClassifier(hidden_layer_sizes=(100,),
                        max_iter=2000,
                        activation='relu',
                        solver='sgd',
                        alpha=0.0001,
                        verbose=1,
                        tol=1e-7,
                        learning_rate_init=0.005,
                        early_stopping=False,
                        )
    mlp.fit(X_train, y_train)

    # Save the trained model and scaler
    # Save the model and scaler
    joblib.dump(mlp, 'mlp_model_sklearn.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Model and scaler saved.")

    accuracy = mlp.score(X_test, y_test)
    print(f"Test Akurasi: {accuracy}")

    result = ""

    if accuracy != 0:
        result = f"Training Successfully.\nAccuracy: {accuracy}"

    return result
