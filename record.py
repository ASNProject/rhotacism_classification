import pyaudio
import wave

def record(callback):
    chunk = 4096  # Record in larger chunks of 4096 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1  # Change this to 1 or 2 depending on your device's capability
    fs = 44100  # Record at 44100 samples per second
    seconds = 10  # Durasi rekaman
    filename = "output.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    print('Recording')

    # Try to open the stream
    try:
        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)
    except OSError as e:
        print(f"Could not open stream: {e}")
        p.terminate()
        exit()

    frames = []  # Initialize array to store frames

    # Store data in chunks for the specified duration
    for i in range(0, int(fs / chunk * seconds)):
        try:
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
            callback(data)  # Panggil callback untuk memperbarui waveform
        except IOError as e:
            print(f"Error recording: {e}")
            # Attempt to reopen the stream if it closes due to overflow
            try:
                stream.stop_stream()
                stream.close()
                stream = p.open(format=sample_format,
                                channels=channels,
                                rate=fs,
                                frames_per_buffer=chunk,
                                input=True)
            except OSError as e:
                print(f"Could not reopen stream: {e}")
                p.terminate()
                exit()

    # Check if the stream is still open before attempting to close it
    if stream.is_active():
        stream.stop_stream()
        stream.close()

    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
