import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def feature_extraction(file_path):
    # Buka data audio
    x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    # Ekstraksi fitur dari audio
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=16).T, axis=0)
    return mfcc


# List untuk menyimpan fitur dari masing-masing audio
features_list_c = []
features_list_tc = []
directory = 'audio_data/'

# Iterasi melalui setiap file audio di direktori
for audio in os.listdir(directory):
    audio_path = os.path.join(directory, audio)  # Membuat path lengkap untuk setiap file audio
    if audio_path.endswith('.wav'):  # Memastikan hanya file audio yang diolah
        mfcc = feature_extraction(audio_path)
        if audio.startswith('c'):
            features_list_c.append([audio] + mfcc.tolist())  # Menyimpan nama file dan fitur sebagai list
        elif audio.startswith('tc'):
            features_list_tc.append([audio] + mfcc.tolist())  # Menyimpan nama file dan fitur sebagai list

# Plot semua MFCC untuk file audio yang dimulai dengan 'c'
plt.figure(figsize=(12, 6))
for features in features_list_c:
    audio = features[0]
    mfcc = features[1:]
    plt.plot(mfcc, label=audio)

plt.title('MFCC of Audio Files Starting with "Cadel"')
plt.xlabel('MFCC Coefficients')
plt.ylabel('Average Amplitude')
plt.legend()
plt.tight_layout()
plt.savefig('mfcc_cadel_plot.png')
plt.show()

# Plot semua MFCC untuk file audio yang dimulai dengan 'tc'
plt.figure(figsize=(12, 6))
for features in features_list_tc:
    audio = features[0]
    mfcc = features[1:]
    plt.plot(mfcc, label=audio)

plt.title('MFCC of Audio Files Starting with "Tidak Cadel"')
plt.xlabel('MFCC Coefficients')
plt.ylabel('Average Amplitude')
plt.legend()
plt.tight_layout()
plt.savefig('mfcc_tidak_cadel_plot.png')
plt.show()

# Konversi list ke DataFrame
columns = ['filename'] + [f'mfcc_{i}' for i in range(16)]
features_df_c = pd.DataFrame(features_list_c, columns=columns)
features_df_tc = pd.DataFrame(features_list_tc, columns=columns)

# Simpan DataFrame ke CSV
features_df_c.to_csv('audio_features_c.csv', index=False)
features_df_tc.to_csv('audio_features_tc.csv', index=False)

# Simpan DataFrame ke Excel
features_df_c.to_excel('audio_features_c.xlsx', index=False)
features_df_tc.to_excel('audio_features_tc.xlsx', index=False)

print("Features saved to 'audio_features_c.csv' and 'audio_features_tc.csv'")
print("Features saved to 'audio_features_c.xlsx' and 'audio_features_tc.xlsx'")
