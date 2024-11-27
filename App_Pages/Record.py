import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import io
from PIL import Image
import requests

# Load your model
model = load_model('model/best_model.keras')

# List of Baybayin class labels (syllables)
class_labels = [
    'a', 'ba', 'be', 'bi', 'bo', 'bu', 'da', 'de', 'di', 'do', 'du', 'e',
    'ga', 'ge', 'gi', 'go', 'gu', 'ha', 'he', 'hi', 'ho', 'hu', 'i', 'ka',
    'ke', 'ki', 'ko', 'ku', 'la', 'le', 'li', 'lo', 'lu', 'ma', 'me', 'mi',
    'mo', 'mu', 'na', 'ne', 'nga', 'nge', 'ngi', 'ngo', 'ngu', 'ni', 'no',
    'nu', 'o', 'pa', 'pe','pi', 'po', 'pu', 'ra', 're', 'ri', 'ro', 'ru', 'sa',
    'se', 'si', 'so', 'su', 'ta', 'te', 'ti', 'to', 'tu', 'u', 'wa', 'we',
    'wi', 'wo', 'wu', 'ya', 'ye', 'yi', 'yo', 'yu'
]

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

}


def extract_voiced_audio(audio_file, target_length=8000):
    # Load audio file with a fixed sample rate (8000 Hz)
    y, sr = librosa.load(audio_file, sr=8000)

    # Use librosa's split function to isolate voiced segments
    voiced_segments = librosa.effects.split(y, top_db=20)

    # Concatenate the voiced segments
    voiced_audio = np.concatenate([y[start:end] for start, end in voiced_segments])

    # Pad or truncate to the target length (8000 samples)
    if len(voiced_audio) < target_length:
        voiced_audio = np.pad(voiced_audio, (0, target_length - len(voiced_audio)), mode='constant')
    elif len(voiced_audio) > target_length:
        voiced_audio = voiced_audio[:target_length]

    # Reshape to match model input shape (8000, 1)
    voiced_audio = voiced_audio.reshape((target_length, 1))

    # Expand dimensions to include batch size (1, 8000, 1)
    features = np.expand_dims(voiced_audio, axis=0)

    return features


def preprocess_audio(audio_file):
    # Process audio to extract features (raw audio here)
    audio_data = extract_voiced_audio(audio_file)
    return audio_data


def predict_baybayin(audio_data):

    audio_data_tensor = tf.convert_to_tensor(audio_data)
    prediction = model.predict(audio_data_tensor)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    prediction_prob = prediction[0][predicted_class_index]  # Confidence score of the prediction

    # Map to Baybayin label
    predicted_label = class_labels[predicted_class_index]
    return predicted_label, prediction_prob  # Return label and confidence


def app():
    # Streamlit App UI
    st.title("Speech-to-Baybayin Transcription App")
    st.write("Upload an audio file for transcription:")

    # Custom CSS to resize the slider
    st.markdown("""
    <style>
        .stSlider > div > div > div {
            width: 250px !important;  /* Set a smaller width for the slider */
            margin: auto;  /* Center the slider */
        }
        .download-btn-container {
            margin-top: 20px;  /* Adjust the space between image and download button */
        }
    </style>
    """, unsafe_allow_html=True)

    # Upload an audio file
    audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])

    if audio_file is not None:
        # Display the uploaded audio file
        

        # Preprocess the audio file
        audio_data = preprocess_audio(audio_file)

        # Predict the corresponding Baybayin label and confidence
        baybayin_label, prediction_prob = predict_baybayin(audio_data)

        # Display the result with prediction percentage
        st.write(f"Predicted Baybayin: **{baybayin_label}** ({prediction_prob * 100:.2f}%)")

        if baybayin_label in baybayin_images:
            baybayin_image_html = baybayin_images[baybayin_label]

            # Set a slider for resizing
            image_width = st.slider("Resize Image Width:", min_value=50, max_value=400, value=120, label_visibility = "collapsed")

            # Display the clickable Baybayin image centered with a white box background
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

            # Fetch and resize the image dynamically based on the slider value
            image_url = baybayin_image_html.split('src="')[1].split('"')[0]
            image_data = requests.get(image_url).content
            image = Image.open(io.BytesIO(image_data))

            # Resize the image according to the slider value
            resized_image = image.resize((image_width, int(image.height * image_width / image.width)))

            # Add a white background and handle transparency
            final_image = Image.new("RGBA", resized_image.size, (255, 255, 255, 255))  # White RGBA background
            final_image.paste(resized_image, (0, 0),
                              resized_image.convert("RGBA"))  # Paste the image with alpha channel
            final_image = final_image.convert("RGB")  # Remove alpha channel for final output

            # Save the final image to a temporary buffer
            img_buffer = io.BytesIO()
            final_image.save(img_buffer, format="PNG")
            img_buffer.seek(0)

            # Create a container for the download button with additional margin
            st.markdown('<div class="download-btn-container">', unsafe_allow_html=True)
            st.download_button(
                label="Download",
                data=img_buffer,
                file_name=f"{baybayin_label}.png",
                mime="image/png"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.write("Image not available for this Baybayin syllable.")



if __name__ == "__main__":
    app()
