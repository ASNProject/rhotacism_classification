import pyaudio
import os
import wave
import librosa
import librosa.display
import numpy as np
from sys import byteorder
from array import array
from struct import pack
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import joblib

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 30


def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD


def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r


def trim(snd_data):
    "Trim the blank spots at the start and end"

    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i) > THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data


def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds * RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds * RATE))])
    return r


def record():
    """
    Record a word or words from the microphone and
    return the data as an array of signed shorts.
    Normalizes the audio, trims silence from the
    start and end, and pads with 0.5 seconds of
    blank sound to make sure VLC et al can play
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
                    input=True, output=True,
                    frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r


def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h' * len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
        result = np.hstack((result, tonnetz))

    # expected_shape = 128
    # if result.shape[0] < expected_shape:
    #     result = np.pad(result, (0, expected_shape - result.shape[0]), 'constant')
    # elif result.shape[0] > expected_shape:
    #     result = result[:expected_shape]
    return result


# Fungsi untuk menampilkan spektrogram dari file audio
def plot_spectrogram(file_name):
    X, sample_rate = librosa.load(file_name)

    # Membuat spektrogram
    plt.figure(figsize=(10, 6))
    plt.specgram(X, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap='inferno', sides='default', mode='default', scale='dB');
    plt.axis('off')

    # Menampilkan spektrogram
    plt.savefig('spectrogram.png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()


def plot_frame_blocking(y, sr, frame_length=0.025, frame_stride=0.01):
    frame_size = int(frame_length * sr)
    frame_step = int(frame_stride * sr)
    signal_length = len(y)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_size)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_size
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(y, z)

    indices = np.tile(np.arange(0, frame_size), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_size, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    plt.figure(figsize=(10, 6))
    plt.plot(frames.T)
    plt.title("Frame Blocking")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()


def plot_pre_emphasis(y, pre_emphasis=0.95):
    emphasized_signal = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    plt.figure(figsize=(10, 6))
    plt.plot(emphasized_signal)
    plt.title("Pre-emphasis")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()


def plot_filter_bank(y, sr, nfilt=20):
    emphasized_signal = np.append(y[0], y[1:] - 0.97 * y[:-1])
    frame_size = 0.025
    frame_stride = 0.01
    frame_length = int(round(frame_size * sr))
    frame_step = int(round(frame_stride * sr))
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz_points / sr)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    plt.figure(figsize=(10, 6))
    plt.imshow(filter_banks.T, cmap='viridis', aspect='auto', origin='lower')
    plt.title("Filter Bank")
    plt.xlabel("Frames")
    plt.ylabel("Filter Banks")
    plt.colorbar(format='%+2.0f dB')
    plt.show()


def plot_mfcc(y, sr, n_mfcc=13):
    emphasized_signal = np.append(y[0], y[1:] - 0.97 * y[:-1])
    mfccs = librosa.feature.mfcc(y=emphasized_signal, sr=sr, n_mfcc=n_mfcc)

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.title("MFCC")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    # load the saved model (after training)
    # model = pickle.load(open("result/mlp_classifier.model", "rb"))
    from utils import load_data, split_data, create_model
    import argparse

    parser = argparse.ArgumentParser(description="""Gender recognition script, this will load the model you trained, 
                                    and perform inference on a sample you provide (either using your voice or a file)""")
    parser.add_argument("-f", "--file", help="The path to the file, preferred to be in WAV format")
    args = parser.parse_args()

    ### GANTI DISINI
    file = 'cv-invalid/tc1.wav'
    # construct the model
    model = create_model()
    # load the saved/trained weights
    model.load_weights("results/model.keras")

    # Menampilkan spektogran
    # plot_spectrogram(file)

    # Menampilkan plot gelombang suara
    y, sr = librosa.load(file)

    # print("Plotting frame blocking...")
    # plot_frame_blocking(y, sr)
    #
    # print("Plotting pre-emphasis...")
    # plot_pre_emphasis(y)
    #
    # print("Plotting filter bank...")
    # plot_filter_bank(y, sr)
    #
    # print("Plotting MFCC...")
    # plot_mfcc(y, sr)
    #
    # print("All plots are generated.")

    if not file or not os.path.isfile(file):
        # if file not provided, or it doesn't exist, use your voice
        print("Please waiting")
        # put the file name here
        file = "cv-invalid/c7.wav"
        # record the file (start talking)
        record_to_file(file)
    # extract features and reshape it
    features = extract_feature(file, mel=True).reshape(1, -1)
    # predict the gender!
    rhotacism_prob = model.predict(features)[0][0]
    normal_prob = 1 - rhotacism_prob
    gender = "rhotacism" if rhotacism_prob > normal_prob else "normal"
    # show the result!
    print("Result:", gender)
    print(f"Probabilities:     Rhotacism: {rhotacism_prob * 100:.2f}%    Normal: {normal_prob * 100:.2f}%")
    # Membuat plot gelombang suara
    plt.figure(figsize=(10, 4))
    plt.plot(y)
    plt.title('Gelombang Suara')
    plt.xlabel('Waktu (sampel)')
    plt.ylabel('Amplitudo')
    plt.show()


def predict_rhotacism():
    from utils import load_data, split_data, create_model
    import argparse

    parser = argparse.ArgumentParser(description="""Gender recognition script, this will load the model you trained, 
                                       and perform inference on a sample you provide (either using your voice or a file)""")
    parser.add_argument("-f", "--file", help="The path to the file, preferred to be in WAV format")
    args = parser.parse_args()

    file = 'output.wav'
    # construct the model
    model = create_model()
    # load the saved/trained weights
    model.load_weights("results/model.keras")

    y, sr = librosa.load(file)

    if not file or not os.path.isfile(file):
        # if file not provided, or it doesn't exist, use your voice
        print("Please waiting")
        # put the file name here
        file = "cv-invalid/c7.wav"
        # record the file (start talking)
        record_to_file(file)
    # extract features and reshape it
    features = extract_feature(file, mel=True).reshape(1, -1)
    # predict the gender!
    rhotacism_prob = model.predict(features)[0][0]
    normal_prob = 1 - rhotacism_prob
    gender = "Cadel" if rhotacism_prob > normal_prob else "Normal"
    # show the result!
    print("Result:", gender)
    print(f"Probabilities:     Rhotacism: {rhotacism_prob * 100:.2f}%    Normal: {normal_prob * 100:.2f}%")
    result_text = f"Result: {gender}\nProb: Rhotacism: {rhotacism_prob * 100:.2f}% Normal: {normal_prob * 100:.2f}%"
    # Membuat plot gelombang suara
    plt.figure(figsize=(10, 4))
    plt.plot(y)
    plt.title('Waveform Result')
    plt.xlabel('Time (sampel)')
    plt.ylabel('Amplitude')
    plt.savefig("image.png")
    # plt.show()
    return gender, rhotacism_prob, normal_prob


# Function for feature extraction from new audio data
# def feature_extraction(file_path):
#     x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
#     mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=50).T, axis=0)
#     return mfcc
#
#
# def prediction_mlp():
#     # Load model
#     model = tf.keras.models.load_model('mlp_model.keras')
#
#     # Load scaler
#     scaler = StandardScaler()
#
#     # Load or define X_train and y_train from your training data
#     features_c = np.load('audio_features_c.npy')
#     features_tc = np.load('audio_features_tc.npy')
#
#     # Create DataFrames from numpy arrays
#     columns = ['filename'] + [f'mfcc_{i}' for i in range(50)]
#     df_c = pd.DataFrame(features_c, columns=columns)
#     df_tc = pd.DataFrame(features_tc, columns=columns)
#
#     # Add labels to DataFrames
#     df_c['label'] = 'cadel'
#     df_tc['label'] = 'tidak_cadel'
#
#     # Concatenate dataframes
#     df = pd.concat([df_c, df_tc], ignore_index=True)
#
#     # Prepare data for training
#     X = df.iloc[:, 1:-1].values  # Features (excluding 'filename' and 'label' columns)
#     y = df['label'].map({'cadel': 0, 'tidak_cadel': 1}).values  # Labels
#
#     # Fit scaler on training data
#     scaler.fit(X)
#
#     # Example new audio file
#     new_audio_file = 'output.wav'
#
#     # Extract features from new audio data
#     new_features = feature_extraction(new_audio_file)
#
#     # Print extracted features
#     print("Extracted MFCC features from new audio data:")
#     for i, value in enumerate(new_features, start=1):
#         print(f"MFCC_{i}: {value}")
#
#     # Reshape new features
#     new_features_reshaped = new_features.reshape(1, -1)
#
#     # Normalize new features
#     new_features_scaled = scaler.transform(new_features_reshaped)
#
#     # Perform prediction
#     prediction = model.predict(new_features_scaled)
#     probability_normal = prediction[0][0]  # Probability of the positive class (tidak_cadel)
#     probability_cadel = 1 - probability_normal  # Probability of the negative class (cadel)
#
#     # Output prediction results and percentages
#     print(f"Probability of 'tidak cadel': {probability_normal * 100:.2f}%")
#     print(f"Probability of 'cadel': {probability_cadel * 100:.2f}%")
#
#     # Determine status based on the threshold
#     if probability_normal >= 0.5:
#         print(f"Audio tidak cadel (Probability: {probability_normal * 100:.2f}%)")
#         status = "Normal"
#     else:
#         print(f"Audio cadel (Probability: {probability_cadel * 100:.2f}%)")
#         status = "Cadel"
#
#     # Visualize the MFCC features
#     plt.figure(figsize=(12, 6))
#     plt.plot(new_features, label='MFCC')
#     plt.title('Average Amplitude of MFCC Features of New Audio Data')
#     plt.xlabel('MFCC Coefficients')
#     plt.ylabel('Average Amplitude')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('image.png')
#     # plt.show()
#     return status, probability_cadel, probability_normal


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
