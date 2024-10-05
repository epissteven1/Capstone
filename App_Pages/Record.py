import os

import cv2
import numpy as np
import speech_recognition as sr
from PIL import Image
import streamlit as st

# Mapping from syllable to image filename
baybayin_image_mapping = {
    'a': 'A.png', 'e': 'E.png', 'i': 'I.png', 'o': 'O.png', 'u': 'U.png',
    'ka': 'ka.png', 'ga': 'ga.png', 'nga': 'nga.png', 'ta': 'ta.png', 'da': 'da.png',
    'na': 'na.png', 'pa': 'pa.png', 'ba': 'Ba.png', 'ma': 'ma.png', 'ya': 'ya.png',
    'ra': 'ra.png', 'la': 'la.png', 'wa': 'wa.png', 'sa': 'sa.png', 'ha': 'ha.png'
}


def audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data, language='tl-PH')  # Tagalog
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results; {e}"


def split_into_syllables(word):
    vowels = 'aeiou'
    syllables = []
    current_syllable = ""
    for char in word:
        if char in vowels:
            current_syllable += char
            syllables.append(current_syllable)
            current_syllable = ""
        else:
            if current_syllable:
                syllables.append(current_syllable)
            current_syllable = char
    if current_syllable:
        syllables.append(current_syllable)
    return syllables


def text_to_baybayin_images(text):
    words = text.split()
    baybayin_images = []
    for word in words:
        syllables = split_into_syllables(word)
        for syllable in syllables:
            image_filename = baybayin_image_mapping.get(syllable)
            if image_filename:
                baybayin_images.append(image_filename)
    return baybayin_images


def render_images_to_image(baybayin_images, output_file, image_dir='App/Image', padding=20):
    images = []
    # Ensure the image_dir is an absolute path
    image_dir = os.path.abspath(image_dir)
    for img_name in baybayin_images:
        img_path = os.path.join(image_dir, img_name)
        print(f"Attempting to load image: {img_path}")  # Debug statement
        try:
            img = Image.open(img_path).convert("RGBA")
            # Convert PIL image to OpenCV format
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)

            # Remove background (assume white background)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            b, g, r, _ = cv2.split(img_cv)
            img_cv = cv2.merge([b, g, r, alpha])

            # Apply Canny edge detection
            edges = cv2.Canny(gray, 100, 200)
            img_cv[:, :, 3] = edges  # Add edges to the alpha channel

            # Convert back to PIL image
            img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA))

            images.append(img)
        except FileNotFoundError:
            st.error(f"Image file '{img_name}' not found in directory '{image_dir}'.")
            print(f"FileNotFoundError: Image file '{img_name}' not found in directory '{image_dir}'.")
        except Exception as e:
            st.error(f"Error loading image '{img_name}': {e}")
            print(f"Exception: Error loading image '{img_name}': {e}")

    if not images:
        st.error("No valid images were loaded to create the output image.")
        return None

    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create a combined image with transparency
    combined_image = Image.new('RGBA', (total_width, max_height), (255, 255, 255, 0))
    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, 0), img)
        x_offset += img.width

    background_width = total_width + 2 * padding
    background_height = max_height + 2 * padding
    background = Image.new('RGBA', (background_width, background_height), (255, 255, 255, 0))

    # Center the combined image on the background
    x_centered = (background_width - total_width) // 2
    y_centered = (background_height - max_height) // 2
    background.paste(combined_image, (x_centered, y_centered), combined_image)

    try:
        background.save(output_file)
        print(f"Output image saved as: {output_file}")  # Debug statement
        return background
    except Exception as e:
        st.error(f"Error saving output image: {e}")
        print(f"Exception: Error saving output image: {e}")
        return None


def app():
    st.title("Baybayin Transcription from Audio")

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        temp_audio_file = "temp_audio_file.wav"
        with open(temp_audio_file, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Transcribe the audio file to text
        text = audio_to_text(temp_audio_file)
        st.write(f"Transcribed Text: {text}")

        # Convert text to Baybayin images
        baybayin_images = text_to_baybayin_images(text)
        if baybayin_images:
            # Render the images into a single image
            combined_image = render_images_to_image(baybayin_images, 'output_image.png', image_dir='Image')
            

            # Display the final image
            st.image(combined_image, caption='Baybayin Transcription')
        else:
            st.write("No Baybayin images found for the transcribed text.")


if __name__ == "__main__":
    app()
