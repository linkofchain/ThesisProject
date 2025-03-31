import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

gemini_api_key = os.getenv("GEMINI_API_KEY")

