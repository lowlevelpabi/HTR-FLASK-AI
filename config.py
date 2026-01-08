import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
USER_SESSIONS_PATH = os.path.join(BASE_DIR, "user_sessions.json")
MODEL_PATH = os.path.join(BASE_DIR, "maruf_89d898f0-581c-4981-b8e9-7c4db1097590.h5")
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")
SCALER_PATH = os.path.join(BASE_DIR, "maruf_62fc92d4-a74e-4ada-b3e1-239aa6261687.pkl")

# Ollama Config
OLLAMA_MODEL_NAME = "gemma3:1b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Defaults
DEFAULT_VALUES = {
    "age": 23,
    "weight": 40.0,
    "gender": 0,
    "activity": 1,
    "humidity": 72,
    "temperature": 30,
    "complication": 0,
    "is_indoors": 1,
    "is_ground_wet": 0,
    "is_windy_or_fanned": 1,
    "is_direct_sun": 0,
    "humidity_scale": 3
}

# Feature Lists
REQUIRED_FEATURES = [
    "age",
    "gender",
    "weight",
    "activity",
    "sub_activity",
    "humidity_scale",
    "temperature",
    "complication",
    "is_indoors",
    "is_ground_wet",
    "is_windy_or_fanned",
    "is_direct_sun",
]

# Mappings
GENDER_MAP = {"male": 1, "female": 0}
ACTIVITY_MAP = {"low": 0, "medium": 1, "high": 2}
COMPLICATION_MAP = {"none": 0, "mild": 1, "severe": 2}
INDOORS_MAP = {"no": 0, "indoors": 1, "outdoors": 0}
WET_GROUND_MAP = {"no": 0, "yes": 1}
BINARY_MAP = {"no": 0, "yes": 1}

# Reverse Mappings
GENDER_MAP_REVERSE = {v: k for k, v in GENDER_MAP.items()}
ACTIVITY_MAP_REVERSE = {v: k for k, v in ACTIVITY_MAP.items()}
COMPLICATION_MAP_REVERSE = {v: k for k, v in COMPLICATION_MAP.items()}

STANDARD_GLASS_ML = 250
