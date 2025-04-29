import streamlit as st
import whisper
import tempfile
import os
import base64
import numpy as np

# Page configuration
st.set_page_config(page_title="Speech Utility App", page_icon="🎙️")

# Cache the Whisper model to load it only once
@st.cache_resource
def load_whisper_model():
    """Load Whisper model with caching to improve performance."""
    st.info("Loading Whisper model...")
    return whisper.load_model("tiny")

def transcribe_uploaded_audio(model, audio_file):
    """Transcribe uploaded audio using Whisper model."""
    if audio_file is None:
        st.warning("No audio uploaded. Please upload an audio file.")
        return ""
    
    try:
        st.info("⏳ Transcribing...")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_filepath = tmp_file.name
        
        # Transcribe the audio file
        result = model.transcribe(tmp_filepath)
        text = result["text"].strip()
        
        # Remove the temporary file
        os.unlink(tmp_filepath)
        
        st.write(f"📝 Transcribed Text: {text}")
        return text
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return ""

def text_to_speech_info():
    """Display information about TTS limitations."""
    st.info("""
    ℹ️ Note: Text-to-speech functionality is not available in the web version.
    For a full experience with speech output, please run this app locally.
    """)

def get_audio_player_html(text):
    """Returns HTML for an audio player that speaks the given text using browser's TTS."""
    html = f"""
    <script>
        function speakText() {{
            const text = "{text}";
            const utterance = new SpeechSynthesisUtterance(text);
            window.speechSynthesis.speak(utterance);
        }}
    </script>
    <button onclick="speakText()" style="background-color:#4CAF50;color:white;padding:10px;border:none;border-radius:5px;cursor:pointer;">
        🔊 Speak Text (Browser TTS)
    </button>
    """
    return html

def main():
    # Initialize session state
    if 'whisper_model' not in st.session_state:
        st.session_state.whisper_model = load_whisper_model()
    
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ""

    # App title and description
    st.title("🎙️ Speech Utility App")
    st.markdown("""
    This app allows you to transcribe speech from audio files using OpenAI's Whisper model.
    Upload an audio file and get the transcribed text.
    """)
    
    # Display info about web limitations
    st.warning("""
    ⚠️ The web version has limited functionality:
    - Audio recording from microphone isn't available
    - Server-side speech synthesis isn't available
    - You can upload audio files for transcription
    """)

    # Main app functionality
    tab1, tab2 = st.tabs(["Audio Transcription", "Text-to-Speech"])

    with tab1:
        st.header("Audio Transcription")
        uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'ogg', 'm4a'])
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
            
            if st.button("🔍 Transcribe Audio", key="transcribe_upload"):
                transcribed_text = transcribe_uploaded_audio(st.session_state.whisper_model, uploaded_file)
                st.session_state.transcribed_text = transcribed_text
                
                # Display transcribed text in a text area for easy copying
                if transcribed_text:
                    st.text_area("Copy transcribed text:", transcribed_text, height=150, key="output_text")

    with tab2:
        st.header("Text-to-Speech")
        # Text input with key to maintain state
        user_text = st.text_area("Enter text to speak:", key="tts_input")
        
        if user_text:
            # Display info about browser-based TTS
            st.markdown("Since server-side TTS isn't available in the web version, you can use your browser's built-in speech synthesis:")
            st.components.v1.html(get_audio_player_html(user_text), height=50)
            
            # Info about word count
            word_count = len(user_text.split())
            st.info(f"Word count: {word_count}")

    # Footer
    st.markdown("---")
    st.markdown("Powered by OpenAI Whisper | Made with Streamlit")
    
if __name__ == "__main__":
    main()