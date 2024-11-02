import os
import speech_recognition as sr
from PIL import Image
import streamlit as st
import tempfile
import soundfile as sf
import base64
import numpy as np
import librosa
import noisereduce as nr
from tensorflow.keras.models import load_model
from io import BytesIO

# Load the Keras model
model = load_model('model/best_model.keras')

# Mapping from syllable to image filename
baybayin_image_mapping = {
    'a': 'A.png', 'e': 'E.png', 'i': 'I.png', 'o': 'O.png', 'u': 'U.png',
    'ka': 'Ka.png', 'ga': 'Ga.png', 'nga': 'Nga.png', 'ta': 'Ta.png', 'da': 'Da.png',
    'na': 'Na.png', 'pa': 'Pa.png', 'ba': 'Ba.png', 'ma': 'Ma.png', 'ya': 'Ya.png',
    'ra': 'Ra.png', 'la': 'La.png', 'wa': 'Wa.png', 'sa': 'Sa.png', 'ha': 'Ha.png',
    'be': 'Be.png', 'bi': 'Bi.png', 'bo': 'Bo.png', 'bu': 'Bu.png', 'de': 'De.png',
    'di': 'Di.png', 'do': 'Do.png', 'du': 'Du.png', 'ge': 'Ge.png', 'gi': 'Gi.png',
    'go': 'Go.png', 'gu': 'Gu.png', 'he': 'He.png', 'hi': 'Hi.png', 'ho': 'Ho.png',
    'hu': 'Hu.png', 'ke': 'Ke.png', 'ki': 'Ki.png', 'ko': 'Ko.png', 'ku': 'Ku.png',
    'le': 'Le.png', 'li': 'Li.png', 'lo': 'Lo.png', 'lu': 'Lu.png', 'me': 'Me.png',
    'mi': 'Mi.png', 'mo': 'Mo.png', 'mu': 'Mu.png', 'ne': 'Ne.png', 'ni': 'Ni.png',
    'no': 'No.png', 'nu': 'Nu.png', 'nge': 'Nge.png', 'ngi': 'Ngi.png', 'ngo': 'Ngo.png',
    'ngu': 'Ngu.png', 'pe': 'Pe.png', 'pi': 'Pi.png', 'po': 'Po.png', 'pu': 'Pu.png',
    're': 'Re.png', 'ri': 'Ri.png', 'ro': 'Ro.png', 'ru': 'Ru.png', 'se': 'Se.png', 'si': 'Si.png',
    'so': 'So.png', 'su': 'Su.png', 'te': 'Te.png', 'ti': 'Ti.png', 'to': 'To.png',
    'tu': 'Tu.png', 'we': 'We.png', 'wi': 'Wi.png', 'wo': 'Wo.png', 'wu': 'Wu.png',
    'ye': 'Ye.png', 'yi': 'Yi.png', 'yo': 'Yo.png', 'yu': 'Yu.png'
}


def reduce_noise(audio_data):
    audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16)
    reduced_noise = nr.reduce_noise(y=audio_np, sr=audio_data.sample_rate)
    return sr.AudioData(reduced_noise.tobytes(), audio_data.sample_rate, audio_data.sample_width)


def extract_voiced_audio(audio_file, target_length=8000):
    # Load the audio at 8 kHz
    y, sr = librosa.load(audio_file, sr=8000)

    # Apply Voice Activity Detection (VAD) to isolate voiced segments
    voiced_segments = librosa.effects.split(y, top_db=20)  # Adjust top_db as needed

    # Concatenate voiced segments to get the voiced-only audio
    voiced_audio = np.concatenate([y[start:end] for start, end in voiced_segments])

    # Save the voiced audio to a temporary file to play back in Streamlit
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        sf.write(temp_audio_file.name, voiced_audio, sr)  # Save voiced segments as a new audio file
        st.audio(temp_audio_file.name, format="audio/wav")

    # Ensure the length matches the model input shape (8000 samples)
    if len(voiced_audio) < target_length:
        # Pad with zeros if shorter than 8000 samples
        voiced_audio = np.pad(voiced_audio, (0, target_length - len(voiced_audio)), mode='constant')
    elif len(voiced_audio) > target_length:
        # Truncate to exactly 8000 samples if longer
        voiced_audio = voiced_audio[:target_length]

    # Reshape to (8000, 1) for compatibility with model input
    voiced_audio = voiced_audio.reshape((target_length, 1))
    # Add a batch dimension to match model input shape (1, 8000, 1)
    features = np.expand_dims(voiced_audio, axis=0)

    return features


def predict_syllables(features):
    # Predict using the model
    predictions = model.predict(features)

    # Get the predicted label and the corresponding confidence
    predicted_label = np.argmax(predictions, axis=1)[0]
    confidence_score = predictions[0][predicted_label]

    # Map the predicted label to the syllable
    syllable_mapping = {
        0: 'a', 1: 'ba', 2: 'be', 3: 'bi', 4: 'bo',
        5: 'bu', 6: 'da', 7: 'de', 8: 'di', 9: 'do',
        10: 'du', 11: 'e', 12: 'ga', 13: 'ge', 14: 'gi',
        15: 'go', 16: 'gu', 17: 'ha', 18: 'he', 19: 'hi',
        20: 'ho', 21: 'hu', 22: 'i', 23: 'ka', 24: 'ke',
        25: 'ki', 26: 'ko', 27: 'ku', 28: 'la', 29: 'le',
        30: 'li', 31: 'lo', 32: 'lu', 33: 'ma', 34: 'me',
        35: 'mi', 36: 'mo', 37: 'mu', 38: 'na', 39: 'ne',
        40: 'nga', 41: 'nge', 42: 'ngi', 43: 'ngo', 44: 'ngu',
        45: 'ni', 46: 'no', 47: 'nu', 48: 'o', 49: 'pa',
        50: 'pe', 51: 'ni', 52: 'po', 53: 'pu', 54: 'ra',
        55: 're', 56: 'ri', 57: 'ro', 58: 'ru', 59: 'sa',
        60: 'se', 61: 'si', 62: 'so', 63: 'su', 64: 'ta',
        65: 'te', 66: 'ti', 67: 'to', 68: 'tu', 69: 'u',
        70: 'wa', 71: 'we', 72: 'wi', 73: 'wo', 74: 'wu',
        75: 'ya', 76: 'ye', 77: 'yi', 78: 'yo', 79: 'yu'
    }

    # Retrieve the syllable name based on the predicted label
    predicted_syllable = syllable_mapping.get(predicted_label, "unknown")
    return predicted_syllable, confidence_score


def text_to_baybayin_images(predicted_syllable, target_size=(100, 100)):
    image_filename = baybayin_image_mapping.get(predicted_syllable)
    if image_filename:
        image_path = os.path.join('Image', image_filename)
        try:
            img = Image.open(image_path)
            # Resize the image while maintaining aspect ratio
            img.thumbnail(target_size, Image.LANCZOS)
            return img
        except FileNotFoundError:
            st.error(f"Image file '{image_filename}' not found.")
    return None


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def app():
    st.title("Baybayin Translator")
    st.write("Upload an audio file for transcription and translation:")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(uploaded_file.read())
            temp_audio_file = f.name

        predicted_syllable, confidence_score = predict_syllables(features)
        st.write(f"Predicted Syllable: {predicted_syllable}")
        st.write(f"Confidence Score: {confidence_score:.2%}")

        # Display Baybayin image with specified size
        baybayin_image = text_to_baybayin_images(predicted_syllable, target_size=(250, 250))
        if baybayin_image:
            image_base64 = image_to_base64(baybayin_image)
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center; align-items: center;">
                    <img src="data:image/png;base64,{image_base64}" alt="Baybayin Transcription" width="150" />
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.write("Could not find an image for the predicted syllable.")


if __name__ == "__main__":
    app()
