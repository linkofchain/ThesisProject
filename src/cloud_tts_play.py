import os
import io
import uuid
from flask import Flask, request, jsonify
from label_studio_ml.model import LabelStudioMLBase
from google.cloud import texttospeech_v1beta1 as texttospeech
from google.cloud import storage

# --- Google Cloud Configuration ---
# IMPORTANT: Replace with your actual Google Cloud Project ID
GCP_PROJECT_ID = "your-gcp-project-id"
# IMPORTANT: Replace with the name of your Google Cloud Storage bucket
GCS_BUCKET_NAME = "your-audio-storage-bucket"

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
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    print("Google Cloud clients initialized successfully.")
except Exception as e:
    print(f"Error initializing Google Cloud clients: {e}")
    print("Please ensure your GOOGLE_APPLICATION_CREDENTIALS are correctly set and the service account has necessary permissions.")
    # Exit or handle gracefully depending on your application's needs
    tts_client = None
    storage_client = None
    bucket = None

class GoogleCloudTTSModel(LabelStudioMLBase):

    def setup(self):
        """Configure any model parameters here.
        For example, voice settings.
        """
        self.target_voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Standard-C", # Or choose a Wavenet/Neural voice for better quality
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
        )
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
        )
        print("GoogleCloudTTSModel setup complete.")

    def predict(self, tasks, **kwargs):
        """This method is called by Label Studio when a prediction is needed.
        In Community Edition, this is typically triggered upon:
        - Initial task loading (if "Interactive pre-annotation" is enabled)
        - Annotation submission
        - Annotation update
        """
        if not tasks:
            print("No tasks received for prediction.")
            return []

        # Get the current task data (Label Studio always sends a list of tasks)
        task = tasks[0] # Assuming we are processing one task at a time for interactive mode

        print(f"Received task data for prediction: {task}")

        # Extract phoneme input from the task data
        # We need to check both original task data and any existing annotation results
        phoneme_text = task.get('data', {}).get('phoneme_string') # Initial data for the task
        
        # If there are existing annotations (e.g., user is updating),
        # check the latest annotation result for the phoneme input.
        # This is crucial for getting the user's latest input after they modify it.
        latest_phoneme_from_annotation = None
        if 'annotations' in task and task['annotations']:
            # Get the most recent annotation
            latest_annotation = task['annotations'][-1]
            for result in latest_annotation.get('result', []):
                if result.get('from_name') == 'phoneme_input' and result.get('type') == 'text':
                    if result.get('value') and result['value'].get('text'):
                        latest_phoneme_from_annotation = result['value']['text'][0]
                        break
        
        # Use the latest annotation's phoneme if available, otherwise fall back to initial task data
        if latest_phoneme_from_annotation:
            phoneme_text = latest_phoneme_from_annotation
        
        # Fallback if somehow phoneme_text is still not set (e.g. empty string)
        if not phoneme_text or not phoneme_text.strip():
            print("No valid phoneme input found in task data or latest annotation.")
            # If we don't have phonemes, we might clear the audio or do nothing
            # Returning an empty prediction for audio will clear it if it was there
            return [{
                "result": [
                    {
                        "from_name": "audio_output",
                        "to_name": "audio_output",
                        "type": "audio",
                        "value": {
                            "audio": None # Clear the audio
                        },
                    }
                ],
                "score": 0.0 # Indicate no valid prediction
            }]


        print(f"Using phoneme input for TTS: {phoneme_text}")

        if tts_client is None or storage_client is None or bucket is None:
            print("Google Cloud clients not initialized. Cannot perform TTS or storage.")
            return []

        # Construct SSML for phoneme input
        # Ensure the phoneme alphabet matches what you're providing (e.g., "ipa")
        ssml_text = f'<speak><phoneme alphabet="ipa" ph="{phoneme_text}">{phoneme_text}</phoneme></speak>'
        synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)

        try:
            # Call Google Cloud TTS API
            response = tts_client.synthesize_speech(
                input=synthesis_input,
                voice=self.target_voice,
                audio_config=self.audio_config,
            )

            # Generate a unique filename for the audio
            audio_filename = f"tts_output_{uuid.uuid4()}.mp3"
            destination_blob_name = f"tts_audio/{audio_filename}"

            # Upload the audio to Google Cloud Storage
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_string(response.audio_content, content_type="audio/mpeg")
            
            # Make the blob publicly accessible. For production, consider using signed URLs
            # and having your backend return a pre-signed URL to Label Studio.
            blob.make_public()
            audio_gcs_url = blob.public_url
            print(f"Generated audio URL: {audio_gcs_url}")

            # Return the prediction in Label Studio format
            # This updates the <Audio> tag
            return [{
                "result": [
                    {
                        "from_name": "audio_output", # Name of your <Audio> tag
                        "to_name": "audio_output", # Name of your <Audio> tag (usually matches from_name for objects)
                        "type": "audio",
                        "value": {
                            "audio": audio_gcs_url,
                        },
                    }
                ],
                "score": 1.0 # Confidence score
            }]

        except Exception as e:
            print(f"Error during TTS synthesis or GCS upload: {e}")
            # If an error occurs, you might want to clear any existing audio or show an error
            return [{
                "result": [
                    {
                        "from_name": "audio_output",
                        "to_name": "audio_output",
                        "type": "audio",
                        "value": {
                            "audio": None # Clear the audio if an error occurred
                        },
                    }
                ],
                "score": 0.0
            }]

    def fit(self, annotations, **kwargs):
        """Optional: Implement model training if you have a training workflow.
        This method is called when annotations are submitted/updated and "Start model
        training on annotation submission" is enabled in project settings.
        Not directly used for the immediate TTS generation, but could log data.
        """
        print(f"Fit method called with {len(annotations)} annotations (not implemented for this example).")
        # You could, for instance, log the phoneme input and generated audio URL here
        # to a database for later analysis or to track usage.
        return {}