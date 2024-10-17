import os
import streamlit as st
import whisper
from pathlib import Path
import time

# Load the Whisper model
model = whisper.load_model("base")

# Create the folder to save transcription text files
OUTPUT_FOLDER = 'transcriptions'
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# Streamlit App
st.title("Audio Transcription App")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # Save the uploaded audio file temporarily
    file_path = os.path.join(OUTPUT_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"Uploaded file: {uploaded_file.name}")
    
    # Perform transcription using Whisper
    st.write("Transcribing...")
    
    try:
        # Create a progress bar
        progress_bar = st.progress(0)

        # Optionally display a spinner while transcribing
        with st.spinner("Please wait..."):
            # Start the transcription
            transcription = model.transcribe(file_path)
            
            # Simulate progress for the duration of the transcription
            for i in range(100):
                time.sleep(0.05)  # Adjust the sleep duration if necessary
                progress_bar.progress(i + 1)  # Update progress bar
            
            transcription_text = transcription['text']
        
        # Check if transcription text is not empty
        if transcription_text:
            # Display the transcription result
            st.text_area("Transcribed Text", transcription_text, height=200)

            # Save the transcription to a text file with UTF-8 encoding
            transcription_file_path = os.path.join(OUTPUT_FOLDER, uploaded_file.name.rsplit('.', 1)[0] + '.txt')
            with open(transcription_file_path, 'w', encoding='utf-8') as f:
                f.write(transcription_text)
            
            # Provide download button for the transcription
            st.download_button(
                label="Download Transcription",
                data=transcription_text,
                file_name=os.path.basename(transcription_file_path),
                mime="text/plain"
            )
        else:
            st.warning("No text was transcribed.")
    
    except Exception as e:
        st.error(f"An error occurred during transcription: {e}")
