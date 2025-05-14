import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import io
from PIL import Image
import requests
import json

# Load your model
model = load_model('model/Filipino_speech_recognition.keras')

# Load class labels from labels.json
with open('model/labels.json', 'r', encoding='utf-8') as f:
    class_labels = json.load(f)
# Define Baybayin font image URLs for each syllable
baybayin_images = {
    'a': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/YQ/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ba': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/Yg/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'be': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/YmU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'bi': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/YmU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'bo': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/YnU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'bu': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/YnU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'da': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/ZA/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'de': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/ZGU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'di': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/ZGU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'do': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/ZHU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'du': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/ZHU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'e' : '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/RQ/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a> ',
    'ga': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/Zw/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ge': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/Z2k/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'gi': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/Z2k/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'go': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/Z28/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'gu': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/Z28/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ha': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/aA/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'he': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/aGU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'hi': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/aGU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ho': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/aG8/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'hu': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/aG8/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'i': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/SQ/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ka': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/aw/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ke': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/a2U/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ki': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/a2U/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ko': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/a3U/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ku': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/a3U/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'la': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/bA/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'le': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/bGU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'li': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/bGU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'lo': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/bG8/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'lu': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/bG8/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ma': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/bQ/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'me': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/bWU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'mi': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/bWU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'mo': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/bXU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'mu': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/bXU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'na': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/bg/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ne': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/bmk/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ni': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/bmk/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'no': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/bm8/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'nu': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/bm8/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'nga': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/Tg/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'nge': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/Tmk/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ngi': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/Tmk/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ngo': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/Tm8/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ngu': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/Tm8/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'pa': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/cA/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'pe': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/cGU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'pi': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/cGU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'po': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/cG8/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'pu': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/cG8/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ra': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/cg/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    're': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/cmU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ri': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/cmU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ro': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/cm8/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ru': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/cm8/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'sa': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/cw/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'se': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/c2U/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'si': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/c2U/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'so': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/c28/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'su': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/c28/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a> ',
    'ta': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/dA/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'te': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/dGU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ti': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/dGU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'to': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/dG8/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'tu': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/dG8/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'u': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/VQ/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'wa': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/dw/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'we': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/d2U/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'wi': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/d2U/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'wo': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/d28/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'wu': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/d28/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ya': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/eQ/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'ye': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/eWU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'yi': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/eWU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'yo': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/eXU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'yu': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/eXU/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'o': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/VQ/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
    'kamusta': '<a href="https://www.fontspace.com/category/baybayin"><img src="https://see.fontimg.com/api/rf5/ZV3MK/MGVjOThmZGMzMDEyNDU2OGI3NmZlMDBjOTJmZDZhMTEudHRm/VQ/bagwis-baybayin-font-regular.png?r=fs&h=91&w=1250&fg=000000&bg=FFFFFF&tb=1&s=73" alt="Baybayin fonts"></a>',
}



def preprocess_audio(audio_file):
    y, sr = librosa.load(audio_file, sr=8000)
    voiced_segments = librosa.effects.split(y, top_db=20)
    voiced_audio = np.concatenate([y[start:end] for start, end in voiced_segments])

    target_length = 8000
    if len(voiced_audio) < target_length:
        voiced_audio = np.pad(voiced_audio, (0, target_length - len(voiced_audio)), mode='constant')
    else:
        voiced_audio = voiced_audio[:target_length]

    mfcc = librosa.feature.mfcc(y=voiced_audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)
    mfcc_std = np.std(mfcc, axis=1, keepdims=True) + 1e-10
    mfcc = (mfcc - mfcc_mean) / mfcc_std

    if mfcc.shape[1] < 100:
        mfcc = np.pad(mfcc, ((0, 0), (0, 100 - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :100]

    mfcc = np.expand_dims(mfcc, axis=-1)
    mfcc = np.expand_dims(mfcc, axis=0)

    return mfcc


def predict_baybayin(audio_data):
    audio_data_tensor = tf.convert_to_tensor(audio_data, dtype=tf.float32)
    prediction = model.predict(audio_data_tensor)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    prediction_prob = prediction[0][predicted_class_index]
    predicted_label = class_labels[predicted_class_index]
    return predicted_label, prediction_prob


def app():
    st.title("Speech-to-Baybayin Transcription App")
    st.write("Upload an audio file for transcription:")

    st.markdown("""
    <style>
        .stSlider > div > div > div {
            width: 250px !important;
            margin: auto;
        }
        .download-btn-container {
            margin-top: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])

    if audio_file is not None:
        audio_data = preprocess_audio(audio_file)
        baybayin_label, prediction_prob = predict_baybayin(audio_data)

        st.write(f"Predicted: **{baybayin_label}** ({int(prediction_prob * 100)}%)")

        if baybayin_label in baybayin_images:
            baybayin_image_html = baybayin_images[baybayin_label]
            image_width = st.slider("Resize Image Width:", min_value=50, max_value=400, value=120, label_visibility="collapsed")

            st.markdown(
                f"""
                <div style="
                    display: flex;  
                    justify-content: center; 
                    align-items: center; 
                    background-color: white; 
                    padding: 20px; 
                    border-radius: 10px; 
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    width: fit-content;
                    margin: auto;
                ">
                    <img src="{baybayin_image_html.split('src="')[1].split('"')[0]}" 
                        alt="Baybayin fonts" 
                        style="width: {image_width}px; max-width: 100%; height: auto;"
                    >
                </div>
                """,
                unsafe_allow_html=True
            )

            image_url = baybayin_image_html.split('src="')[1].split('"')[0]
            image_data = requests.get(image_url).content
            image = Image.open(io.BytesIO(image_data))
            resized_image = image.resize((image_width, int(image.height * image_width / image.width)))

            final_image = Image.new("RGBA", resized_image.size, (255, 255, 255, 255))
            final_image.paste(resized_image, (0, 0), resized_image.convert("RGBA"))
            final_image = final_image.convert("RGB")

            img_buffer = io.BytesIO()
            final_image.save(img_buffer, format="PNG")
            img_buffer.seek(0)

            st.markdown('<div class="download-btn-container">', unsafe_allow_html=True)
            st.download_button(
                label="Download",
                data=img_buffer,
                file_name=f"{baybayin_label}.png",
                mime="image/png"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.write("Image not available for this label.")


if __name__ == "__main__":
    app()
