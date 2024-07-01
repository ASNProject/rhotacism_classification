import librosa
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def feature_extraction(file_path):
    # Buka data audio
    x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    # Ekstraksi fitur dari audio
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=50).T, axis=0)
    return mfcc


# List untuk menyimpan fitur dari masing-masing audio
features_list = []
directory = 'audio_data/'

# Iterasi melalui setiap file audio di direktori
for audio in os.listdir(directory):
    audio_path = os.path.join(directory, audio)  # Membuat path lengkap untuk setiap file audio
    if audio_path.endswith('.wav'):  # Memastikan hanya file audio yang diolah
        features = feature_extraction(audio_path)
        features_list.append([audio] + features.tolist())  # Menyimpan nama file dan fitur sebagai list

        # Plot line MFCC
        # HILANGKAN COMMENT DIBAWAH JIKA INGIN MENAMPILKAN PLOT MASING-MASING AUDIO
        plt.figure(figsize=(10, 4))
        plt.plot(features, label=audio)
        plt.title(f'MFCC of {audio}')
        plt.xlabel('MFCC Coefficients')
        plt.ylabel('Average Amplitude')
        plt.legend()
        plt.tight_layout()
        plt.savefig('mfcc_cadel_plot.png')
        plt.show()

        # Plot semua MFCC dalam satu grafik dengan warna berbeda
        # HILANGKAN COMMENT DIBAWAH JIKA INGIN MENAMPILKAN PLOT MASING-MASING AUDIO
        # plt.figure(figsize=(12, 6))
        # for features in features_list:
        #     audio = features[0]
        #     mfcc = features[1:]
        #     plt.plot(mfcc, label=audio)
        #
        # plt.title('MFCC of All Audio Files')
        # plt.xlabel('MFCC Coefficients')
        # plt.ylabel('Average Amplitude')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

# Konversi list ke DataFrame
columns = ['filename'] + [f'mfcc_{i}' for i in range(50)]
features_df = pd.DataFrame(features_list, columns=columns)

# Simpan DataFrame ke CSV
features_df.to_csv('audio_features.csv', index=False)

# Simpan DataFrame ke Excel
features_df.to_excel('audio_features.xlsx', index=False)

print("Features saved to 'audio_features.csv' and 'audio_features.xlsx'")
