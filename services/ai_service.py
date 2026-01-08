import ollama
import requests
from config import OLLAMA_BASE_URL, OLLAMA_MODEL_NAME

class AiService:
    def __init__(self):
        self.ollama_client = None
        self.initialize_ollama_client()

    def initialize_ollama_client(self):
        """Initializes the Ollama client and tests connection to the server."""
        print("[STARTING] Initializing Ollama Client...")
        try:
            self.ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
            # Attempt to list models to confirm connection and authentication
            self.ollama_client.list()
            print(f"[SUCCESS] Ollama Client connected successfully to {OLLAMA_BASE_URL}")

        except requests.exceptions.ConnectionError:
            print(f"[FAILED] Error: Could not connect to Ollama server at {OLLAMA_BASE_URL}.")
            print(
                "Please ensure the Ollama application is running and the model is pulled."
            )
            self.ollama_client = None
        except Exception as e:
            print(f"[FAILED-ERROR] Error during Ollama initialization: {e}")
            self.ollama_client = None

    def get_gemma_response(self, user_message, chat_history):
        if not self.ollama_client:
             return "I'm sorry, the AI service is currently unavailable. I can only perform hydration analysis."

        system_prompt = (
            "You are a professional assistant. Use Markdown for all responses: "
            "- Use '---' on a new line to separate numbering or distinct topics/sections or key differences or concepts."
            "- Use bold (**text**) for emphasis. "
            "- Use bulleted lists (- item) for steps or data. "
            "Maintain a clear, spaced-out structure."
        )

        system_message = {
            "role": "system", 
            "content": system_prompt
        }

        messages_payload = [system_message] + chat_history + [{"role": "user", "content": user_message}]

        try:
            response = self.ollama_client.chat(
                model=OLLAMA_MODEL_NAME, 
                messages=messages_payload, 
                options={
                    "temperature": 0.6,
                }
            )

            raw_response = response["message"]["content"]
            return raw_response

        except Exception as e:
            print(f"Error generating Ollama response: {e}")
            return "I'm sorry, I couldn't process that request right now."

ai_service = AiService()
