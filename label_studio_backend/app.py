# https://flask.palletsprojects.com/en/stable/quickstart/
# https://pythonbasics.org/flask-tutorial-hello-world/
from flask import Flask, request, jsonify
import os
import io
import uuid
import requests
from google.cloud import storage, texttospeech
from dotenv import load_dotenv

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
LS_API_TOKEN = os.getenv("LABEL_STUDIO_API_TOKEN")
LS_HOST_URL = os.getenv("LABEL_STUDIO_HOST_URL")

storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

app = Flask(__name__)

# is server running?
@app.route('/')
def hello_world():
    return 'Hello, World! The backend server is running.'

# Function to upload a file to GCS and return a gs:// URI
def upload_and_return_gs_uri(audio_data):
    # Generate a unique filename
    filename = f"tts_audio/{uuid.uuid4()}.wav"
    # blob is just binary object storing unstructured data
    blob = bucket.blob(filename)
    
    # Upload the audio data.
    audio_data.seek(0)
    blob.upload_from_file(audio_data, content_type="audio/wav")
    
    # Construct and return the gs:// URI
    return f"gs://{GCS_BUCKET_NAME}/{filename}"

# TTS Generation Logic
def generate_tts(text_to_convert,word):
    tts_client = texttospeech.TextToSpeechClient()
    # revisit this if ipa format with spaces between causes problems for tts generation            
    ssml_input = f'<speak> <phoneme alphabet="ipa" ph="{text_to_convert}">{word}</phoneme> </speak>'

    synthesis_input = texttospeech.SynthesisInput(ssml=ssml_input)
    voice_params = texttospeech.VoiceSelectionParams(
        language_code='en-US',
        ssml_gender=getattr(texttospeech.SsmlVoiceGender, "FEMALE")
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)

    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice_params,
        audio_config=audio_config
    )
                
    audio_stream = io.BytesIO(response.audio_content)
                
    return audio_stream

# updates the task in Label Studio with the new tts audio URI
def update_label_studio_task(task_id, audio_uri):
    url = f"{LS_HOST_URL}/api/tasks/{task_id}/"
    headers = {
        "Authorization": f"Token {LS_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # The payload to update the task's data field
    payload = {
        "data": {
            "tts_audio": audio_uri
        }
    }
    
    try:
        response = requests.put(url, headers=headers, json=payload)
        response.raise_for_status() # Raises an exception for bad status codes
        print(f"Successfully updated task {task_id} in Label Studio.")
        return True
    except requests.exceptions.HTTPError as err:
        print(f"Failed to update task {task_id}: {err}")
        return False

# webhook logic
@app.route('/webhook', methods=['POST'])
def handle_webhook():
    try:
        payload = request.json
        
        if 'result' in payload['annotation'] and payload['annotation']['result']:
            task_id = payload['annotation']['task']
            modified_ipa = payload['annotation']['result'][0]['value']['text'][0]
            task_word = payload['data'].get('word') 
            
            print(f"Processing Task ID: {task_id}")
            print(f"Extracted IPA: {modified_ipa}")
            print(f"Word: {task_word}")
            
            audio_data_bytes = generate_tts(modified_ipa, task_word)
            # GCS Upload and URI Retrieval
            audio_uri = upload_and_return_gs_uri(audio_data_bytes)
            print(f"Uploaded to GCS. GS URI: {audio_uri}")
            
                        # --- Close the Loop ---
            if update_label_studio_task(task_id, audio_uri):
                return jsonify({'status': 'success', 'message': 'Task updated successfully'})
            else:
                return jsonify({'status': 'error', 'message': 'Failed to update Label Studio task'}), 500

        return jsonify({'status': 'success', 'message': 'No annotation result to process'})


    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Run the app locally, 127.0.0.1.
# The 'host' parameter makes it accessible from the network, which is useful for webhooks.
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)