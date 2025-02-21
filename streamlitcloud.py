import streamlit as st
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from supabase import create_client, Client

# ------------------ Supabase Credentials ------------------ #
# Replace these with your actual Supabase project details
SUPABASE_URL = "https://<YOUR-PROJECT-REF>.supabase.co"
SUPABASE_KEY = "<YOUR-ANON-OR-SERVICE-KEY>"
BUCKET_NAME = "my-audio-bucket"  # The name of your Supabase storage bucket


# ---------------------------------------------------------- #

# Create Supabase client
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


supabase: Client = init_supabase()


def save_wav_and_upload(data, sr, file_name):
    """
    Saves the in-memory audio data as a WAV file,
    then uploads it to Supabase storage.
    """
    local_path = os.path.join("temp_audio", file_name)
    os.makedirs("temp_audio", exist_ok=True)

    # Save locally as WAV
    sf.write(local_path, data, sr)

    # Upload to Supabase storage
    with open(local_path, "rb") as f:
        res = supabase.storage.from_(BUCKET_NAME).upload(f"audio/{file_name}", f)

    # You could also get a public URL if your bucket is public:
    public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(f"audio/{file_name}")
    return local_path, public_url


def generate_spectrogram_and_upload(audio_data, sr, file_name):
    """
    Generates a mel-spectrogram from the audio data, saves as PNG,
    and uploads to Supabase storage.
    """
    # Mel-spectrogram
    S = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Plot to a figure (no display)
    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    plt.title(file_name)
    plt.colorbar(format="%+2.f dB")

    # Save locally
    os.makedirs("temp_spectrograms", exist_ok=True)
    base, _ = os.path.splitext(file_name)
    png_name = f"{base}.png"
    local_path = os.path.join("temp_spectrograms", png_name)
    fig.savefig(local_path)
    plt.close(fig)

    # Upload PNG to Supabase
    with open(local_path, "rb") as f:
        res = supabase.storage.from_(BUCKET_NAME).upload(f"spectrograms/{png_name}", f)

    public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(f"spectrograms/{png_name}")
    return local_path, public_url


def main():
    st.title("Audio Upload + Spectrogram + Supabase Demo")

    st.write("Upload up to 5 audio files (ideally ~1 second each).")

    # We'll collect them in a list
    uploaded_files = []
    for i in range(5):
        file = st.file_uploader(f"Upload file #{i + 1} (optional)", type=["wav", "mp3", "ogg"], key=f"uploader_{i}")
        if file is not None:
            uploaded_files.append(file)

    if st.button("Process & Upload"):
        if not uploaded_files:
            st.warning("No files uploaded!")
            return

        # For each uploaded file, load -> save -> create spectrogram -> upload
        for file in uploaded_files:
            # Convert to a unique file name
            # (Streamlit file_uploader doesn't give a real path, it's an in-memory buffer)
            file_name = file.name
            st.write(f"Processing: {file_name}")

            # Load audio into memory with librosa
            # We pass file.read() bytes to soundfile via a buffer
            data, sr = librosa.load(file, sr=16000, mono=True)  # force 16 kHz if desired

            # Step 1: Save .wav, upload to Supabase
            local_wav_path, audio_url = save_wav_and_upload(data, sr,
                                                            file_name="processed_" + file_name.replace(" ", "_"))
            st.write(f" - Uploaded Audio URL: {audio_url['publicURL'] if audio_url else 'Error'}")

            # Step 2: Generate & upload spectrogram
            local_png_path, png_url = generate_spectrogram_and_upload(data, sr,
                                                                      file_name="processed_" + file_name.replace(" ",
                                                                                                                 "_"))
            st.write(f" - Uploaded Spectrogram URL: {png_url['publicURL'] if png_url else 'Error'}")

        st.success("All uploads complete!")


if __name__ == "__main__":
    main()