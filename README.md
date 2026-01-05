# HTR-FLASK-AI

### 01/05/2026 - Major Update

1. **Ollama Initialization**
   - **Open WebUI**
     - We are utilizing the Open WebUI now to make the local LLM to use or browse internet but it still uses the Ollama

     **Initialization Code Snippet Usage**
     ```
     OLLAMA_MODEL_NAME = "gemma3:1b"
     OLLAMA_BASE_URL = "http://localhost:3000/api" 
     OPEN_WEBUI_API_KEY = "<your_own_api_key_here>"
     ollama_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OPEN_WEBUI_API_KEY)
     ```

     **Response Code Snippet Usage**
     ```
     try:
        response = ollama_client.chat.completions.create(
            model=OLLAMA_MODEL_NAME, 
            messages=messages_payload,
            extra_body={
                "features": {
                    "web_search": True 
                }
            }
        )

        return response.choices[0].message.content
     ```