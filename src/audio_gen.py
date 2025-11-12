import os
import io
import uuid
#from google.cloud import texttospeech

from google.cloud import texttospeech_v1beta1 as texttospeech
from google.cloud import storage
from datasets import load_from_disk

# IMPORTANT: Set this environment variable to the path of your Google Cloud service account key JSON file
# For local development, you might set this directly or through your shell:
# export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
# Ensure the service account has "Cloud Text-to-Speech Synthesizer" and "Storage Object Admin" roles.
try:
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        print("WARNING: GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
        print("Please set it to the path of your Google Cloud service account key JSON file.")
        # As a fallback for local testing, you might hardcode the path, but it's not recommended for production:
        # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your/service-account-key.json"
    
    tts_client = texttospeech.TextToSpeechClient()
    print("Google Cloud TTSclients initialized successfully.")
except Exception as e:
    print(f"Error initializing Google Cloud clients: {e}")
    print("Please ensure your GOOGLE_APPLICATION_CREDENTIALS are correctly set and the service account has necessary permissions.")
    # Exit or handle gracefully depending on your application's needs
    tts_client = None

# google-cloud-texttospeech minimum version 2.29.0 is required.

import os
from google.cloud import texttospeech

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

def synthesize(ssml, output_filepath: str = "output.mp3"):
    """Synthesizes speech from the input text and saves it to an MP3 file.

    Args:
        prompt: Styling instructions on how to synthesize the content in
          the text field.
        text: The text to synthesize.
        output_filepath: The path to save the generated audio file.
          Defaults to "output.mp3".
    """
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(ssml=ssml)

    # Select the voice you want to use.
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="Charon",  # Example voice, adjust as needed
        model_name="gemini-2.5-pro-tts"
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type.
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # The response's audio_content is binary.
    with open(output_filepath, "wb") as out:
        out.write(response.audio_content)
        print(f"Audio content written to file: {output_filepath}")

def main():
    phoneme_text = ""
    synthesize(f'<speak><phoneme alphabet="ipa" ph="{phoneme_text}">{phoneme_text}</phoneme></speak>')

def fetch_mispronounced_list():
    # Load splits with datasets
    dataset_dict = load_from_disk("/kaggle/input/irish-phoneme-data-prep-for-wav2vec2/teangl_phon_dataset")

if __name__ == "__main__":
    main()