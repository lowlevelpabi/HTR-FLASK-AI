import json
import os
from config import USER_SESSIONS_PATH

class SessionService:
    def __init__(self):
        self.sessions = {}
        self.load_sessions()

    def load_sessions(self):
        """Loads user sessions from the JSON file on startup."""
        if os.path.exists(USER_SESSIONS_PATH):
            try:
                with open(USER_SESSIONS_PATH, "r") as f:
                    data = f.read()
                    self.sessions = json.loads(data) if data else {}
                print(f"✅ Loaded {len(self.sessions)} user sessions from disk.")
            except json.JSONDecodeError:
                self.sessions = {}
            except Exception as e:
                print(f"❌ Error loading sessions: {e}")
                self.sessions = {}
        else:
            self.sessions = {}

    def save_sessions(self):
        """Saves active user sessions to the JSON file."""
        sessions_to_save = {
            k: v
            for k, v in self.sessions.items()
            if v.get("last_intent") is not None or v.get("data")
        }
        try:
            with open(USER_SESSIONS_PATH, "w") as f:
                json.dump(sessions_to_save, f, indent=4)
        except Exception as e:
            print(f"❌ Error saving sessions: {e}")

    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "last_intent": None,
                "current_field": None,
                "data": {},
                "chat_history": []
            }
        return self.sessions[session_id]

    def clear_session(self, session_id):
        if session_id in self.sessions:
             self.sessions[session_id]["last_intent"] = None
             self.sessions[session_id]["current_field"] = None

session_service = SessionService()
