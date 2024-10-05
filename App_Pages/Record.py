import speech_recognition as sr
from PIL import Image
import streamlit as st
import os
import base64

# Mapping of Baybayin characters to image filenames
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
    'Mi': 'Mi.png', 'mo': 'Mo.png', 'mu': 'Mu.png', 'ne': 'Ne.png', 'ni': 'Ni.png',
    'no': 'No.png', 'nu': 'Nu.png', 'nge': 'Nge.png', 'ngi': 'Ngi.png', 'ngo': 'Ngo.png',
    'ngu': 'Ngu.png', 'pe': 'Pe.png', 'pi': 'Pi.png', 'po': 'Po.png', 'pu': 'Pu.png',
    're': 'Re.png', 'ri': 'Ri.png', 'ro': 'Ro.png', 'ru': 'Ru.png', 'se': 'Se.png', 'si': 'Si.png',
    'so': 'So.png', 'su': 'Su.png', 'te': 'Te.png', 'ti': 'Ti.png', 'to': 'To.png',
    'tu': 'Tu.png', 'we': 'We.png', 'wi': 'Wi.png', 'wo': 'Wo.png', 'wu': 'Wu.png',
    'ye': 'Ye.png', 'yi': 'Yi.png', 'yo': 'Yo.png', 'yu': 'Yu.png'
}

def audio_to_text(audio_file):
    if not os.path.exists(audio_file):
        return "Audio file not found."
    
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language='tl-PH')
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results; {e}"
    except Exception as e:
        return f"An error occurred: {e}"

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

def render_images_to_image(baybayin_images, output_file, image_dir='capstone/Capstone/Image', padding=20, max_width=800):
    images = []
    image_dir = os.path.abspath(image_dir)
    
    for img_name in baybayin_images:
        img_path = os.path.join(image_dir, img_name)
        print(f"Attempting to load image: {img_path}")
        try:
            img = Image.open(img_path)
            images.append(img)
        except FileNotFoundError:
            st.error(f"Image file '{img_name}' not found in directory '{image_dir}'.")
        except Exception as e:
            st.error(f"Error loading image '{img_name}': {e}")

    if not images:
        st.error("No valid images were loaded to create the output image.")
        return None

    # Resize images dynamically based on the number of images
    num_images = len(images)
    if num_images > 2:
        # Calculate optimal width for each image to fit within max_width
        target_width = max_width // num_images
        resized_images = [img.resize((target_width, int(img.height * (target_width / img.width)))) for img in images]
    else:
        resized_images = images  # No resizing for 2 or fewer images

    # Calculate total width and height of the combined image
    total_width = sum(img.width for img in resized_images) + (padding * (num_images - 1))
    max_height = max(img.height for img in resized_images)

    # Create a new image with background
    combined_image = Image.new('RGB', (total_width, max_height), 'white')

    # Paste each image into the combined image
    x_offset = 0
    for img in resized_images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width + padding

    # Add padding around the combined image
    background_width = total_width + 2 * padding
    background_height = max_height + 2 * padding
    background = Image.new('RGB', (background_width, background_height), 'white')

    background.paste(combined_image, (padding, padding))

    try:
        background.save(output_file)
        print(f"Output image saved as: {output_file}")
        return background
    except Exception as e:
        st.error(f"Error saving output image: {e}")
        return None

def app():
    st.title("Baybayin Transcription from Audio")

    uploaded_file = st.file_uploader("Browse or Record Audio", type=["wav", "mp3"])
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_audio_file", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Convert audio to text
        text = audio_to_text("temp_audio_file")
        st.write(f"Transcribed Text: {text}")

        # Convert text to Baybayin images
        baybayin_images = text_to_baybayin_images(text)

        # Render images to a single image
        combined_image = render_images_to_image(baybayin_images, 'output_image.png')
        if combined_image:
            # Save the combined image to a file
            combined_image_path = 'output_image.png'
            combined_image.save(combined_image_path)

            # Encode the image to base64
            with open(combined_image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()

            # Display the image centered with a specific width
            st.markdown(
                f'<div style="text-align: center;"><img src="data:image/png;base64,{encoded_image}" alt="Baybayin Transcription" style="width: 25%; height: 200px;"></div>',
                unsafe_allow_html=True
            )
        else:
            st.write("No Baybayin images found for the transcribed text.")

if __name__ == "__main__":
    app()
