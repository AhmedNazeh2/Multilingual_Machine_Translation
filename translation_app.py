import streamlit as st
import numpy as np
import time
from langdetect import detect, DetectorFactory  # type: ignore
from typing import Tuple, Optional


# Dictionary mapping language codes to full names
LANGUAGE_DICT = {
    'af': 'Afrikaans', 'ar': 'Arabic', 'bg': 'Bulgarian', 'bn': 'Bengali', 'ca': 'Catalan',
    'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'el': 'Greek',
    'en': 'English', 'es': 'Spanish', 'et': 'Estonian', 'fa': 'Persian', 'fi': 'Finnish',
    'fr': 'French', 'gu': 'Gujarati', 'he': 'Hebrew', 'hi': 'Hindi', 'hr': 'Croatian',
    'hu': 'Hungarian', 'id': 'Indonesian', 'it': 'Italian', 'ja': 'Japanese', 'kn': 'Kannada',
    'ko': 'Korean', 'lt': 'Lithuanian', 'lv': 'Latvian', 'mk': 'Macedonian', 'ml': 'Malayalam',
    'mr': 'Marathi', 'ne': 'Nepali', 'nl': 'Dutch', 'no': 'Norwegian', 'pa': 'Punjabi',
    'pl': 'Polish', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 'sk': 'Slovak',
    'sl': 'Slovenian', 'so': 'Somali', 'sq': 'Albanian', 'sv': 'Swedish', 'sw': 'Swahili',
    'ta': 'Tamil', 'te': 'Telugu', 'th': 'Thai', 'tl': 'Tagalog', 'tr': 'Turkish',
    'uk': 'Ukrainian', 'ur': 'Urdu', 'vi': 'Vietnamese', 'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)'
}

# Set page config at the very beginning
st.set_page_config(page_title="Multilingual Translator", page_icon="🌐", layout="wide")

# Attempt to import required libraries
try:
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
except ImportError:
    st.error("Required libraries are not installed. Please run: pip install transformers")
    st.stop()

# Initialize the model name
model_name = "facebook/m2m100_418M"

@st.cache_resource
def load_model():
    try:
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please check your internet connection and try again. If the problem persists, the model might be temporarily unavailable.")
        return None, None

model, tokenizer = load_model()

if model is None or tokenizer is None:
    st.stop()

def translate(text, src_lang,target_lang):
    try:
        tokenizer.src_lang = src_lang  # Set the detected source language
        encoded_input = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.get_lang_id(target_lang))

        #Decode the translated text
        translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        return translated_text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return ""

def detect_language(text: str) -> Optional[str]:
    try:
        return detect(text)
    except:
        st.error("Unable to detect the language. Please try a longer text or check the input.")
        return None
    
st.title("🌍 Multilingual to English Translator 🌎")

input_text = st.text_area("Enter text to translate:", height=150)

st.markdown("""
<style>
.big-font { font-size:20px !important; }
.green-border { 
    border: 2px solid #28a745;  /* Green border */
    padding: 10px; 
    border-radius: 5px; 
}
</style>
""", unsafe_allow_html=True)

target_lang = 'en'
if st.button("Translate"):
    if input_text:
        detected_lang = detect_language(input_text)
        if detected_lang == target_lang:
            st.info("The text is already in English: ")
            translated_text = input_text

        else:
            st.info(f"Detected language: {LANGUAGE_DICT[detected_lang]}")
        with st.spinner("Translating..."):
            translated_text = translate(input_text, detected_lang,target_lang)

        # Display translation with a green border
        if translated_text:
            st.markdown(
                f'<div class="green-border"><p class="big-font">{translated_text}</p></div>', 
                unsafe_allow_html=True
            )
        else:
            st.warning("Translation failed. Please try again.")
    else:
        st.warning("Please enter some text to translate.")

st.markdown("---")
st.markdown("## 🚀 Features")
st.markdown("- Fast and accurate translation")
st.markdown("- Powered by state-of-the-art multilingual transformer model")
st.markdown("- Supports long text input")

st.markdown("## 📊 Translation Statistics")
col3, col4, col5 = st.columns(3)
with col3:
    st.metric(label="Average Translation Time", value="1-2 seconds")
with col4:
    st.metric(label="Supported Languages", value="50+")
with col5:
    st.metric(label="Model Parameters", value="418M")

st.markdown("---")
st.markdown("## 💡 Did you know?")
fun_facts = [
    "This translator supports over 50 different languages!",
    "The model used for translation has 610 million parameters.",
    "Machine translation has come a long way since its inception in the 1950s.",
    "Neural machine translation, like the one used here, often outperforms traditional statistical methods.",
]
st.info(np.random.choice(fun_facts))

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Hugging Face Transformers")