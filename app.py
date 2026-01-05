from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tf_keras import losses
from tf_keras import metrics
from openai import OpenAI
import json
import random
import time
import os
import re
import joblib

### ---------------------------- THIS IS THE DUPLICATED FLASK API FOR HEALTH TIP RECOMMENDER SYSTEM: HYDRATION VERSION WITH OLLAMA INTEGRATION ---------------------------- ###
####### =================================== FLASK API WITH CUSTOM PREDICTION LOGIC AND MODEL WITH OLLAMA CHAT INTEGRATION LLM ===================================== ###########

"""
This is the Configuration and Section for OLLAMA Client Initialization as well as the Model Name and Base URL.
DO NOT EDIT UNLESS YOU KNOW WHAT YOU ARE DOING.
"""
OLLAMA_MODEL_NAME = "gemma3:1b"
OLLAMA_BASE_URL = "http://localhost:3000/api" 
OPEN_WEBUI_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImNmMmY3ZDYwLWQ3NmQtNDU2Zi1hYWNmLWI5YTFmNDFhOTgwYyIsImV4cCI6MTc2OTk2MDgyNywianRpIjoiNTY3NGU4NDAtOTg2Yy00ZGQ5LWJiZWMtZmNmODdkODIwNjAzIn0._3TdeIzo0milQXKrHNnEfVo4TZVxb0j9Tivds4XSzow"
ollama_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OPEN_WEBUI_API_KEY)

def initialize_ollama_client():
    global ollama_client
    print("[STARTING] Initializing Open WebUI Client...")
    try:
        # Connect to Open WebUI instead of raw Ollama
        ollama_client = OpenAI(
            base_url=OLLAMA_BASE_URL, 
            api_key=OPEN_WEBUI_API_KEY
        )
        print(f"[SUCCESS] Connected to Open WebUI at {OLLAMA_BASE_URL}")
    except Exception as e:
        print(f"[ERROR] Failed to connect: {e}")

initialize_ollama_client()

app = Flask(__name__)
CORS(app)

USER_SESSIONS_PATH = os.path.join(os.path.dirname(__file__), "user_sessions.json") 

MODEL_PATH = os.path.join(os.path.dirname(__file__), "maruf_89d898f0-581c-4981-b8e9-7c4db1097590.h5") 
INTENTS_PATH = os.path.join(os.path.dirname(__file__), "intents.json") 
SCALER_PATH = "maruf_62fc92d4-a74e-4ada-b3e1-239aa6261687.pkl" 

# Global Variables
model = None
intents = {"intents": []}
scaler = None
user_sessions = {}

try:
    # 1. Define custom objects
    custom_objects = {
        "mse": losses.MeanSquaredError(),
        "mae": metrics.MeanAbsoluteError(),
    }

    # 2. Load the model
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)

    # 3. Load other assets
    scaler = joblib.load(SCALER_PATH)
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        intents = json.load(f)
    print("✅ Model, scaler, and intents loaded successfully.")

except Exception as e:
    print(f"❌ Error loading assets: {e}")
    model = None
    intents = {"intents": []}

default_values = {
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
}


# --- Numeric Text Parser
def parse_numeric_text(value):
    """Extract numeric value from a string, return float or None"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    if match:
        return float(match.group())
    return None


# --- HELPER FUNCTION TO FIND MISSING DATA ---
required_features = [
    "age",
    "gender",
    "weight",
    "activity",
    "sub_activity",
    "humidity_scale",   # <--- CHANGED FROM "humidity"
    "temperature",
    "complication",
    "is_indoors",
    "is_ground_wet",
    "is_windy_or_fanned",
    "is_direct_sun",
]

# ... (rest of the file remains the same until the /chat route)

gender_map = {"male": 1, "female": 0}
activity_map = {"low": 0, "medium": 1, "high": 2}
complication_map = {"none": 0, "mild": 1, "severe": 2}
indoors_map = {"no": 0, "indoors": 1, "outdoors": 0}
wet_ground_map = {"no": 0, "yes": 1}
binary_map = {"no": 0, "yes": 1}

gender_map_reverse = {v: k for k, v in gender_map.items()}
activity_map_reverse = {v: k for k, v in activity_map.items()}
complication_map_reverse = {v: k for k, v in complication_map.items()}


def get_first_missing_feature(session_data, local_data):
    """Checks session data + local data for missing required features."""
    all_data = {
        k: v.lower() if isinstance(v, str) else v for k, v in local_data.items()
    }
    all_data.update(session_data)

    for feature in required_features:
        value = all_data.get(feature)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return feature
        if (
            feature in ["age", "weight", "humidity", "temperature"]
            and parse_numeric_text(value) is None
        ):
            return feature
        if feature == "gender" and str(value).lower() not in gender_map:
            return feature
        if feature == "activity" and str(value).lower() not in activity_map:
            return feature
        if feature == "is_indoors" and str(value).lower() not in indoors_map:
            return feature
        if feature == "is_ground_wet" and str(value).lower() not in wet_ground_map:
            return feature
        if (
            feature in ["is_windy_or_fanned", "is_direct_sun"]
            and str(value).lower() not in binary_map
        ):
            return feature

    return None


# -----------------------------
# Session Persistence Functions
# -----------------------------
def load_sessions():
    """Loads user sessions from the JSON file on startup."""
    global user_sessions
    if os.path.exists(USER_SESSIONS_PATH):
        try:
            with open(USER_SESSIONS_PATH, "r") as f:
                data = f.read()
                user_sessions = json.loads(data) if data else {}
            print(f"✅ Loaded {len(user_sessions)} user sessions from disk.")
        except json.JSONDecodeError:
            user_sessions = {}
        except Exception as e:
            user_sessions = {}


def save_sessions():
    """Saves active user sessions to the JSON file."""
    global user_sessions
    sessions_to_save = {
        k: v
        for k, v in user_sessions.items()
        if v.get("last_intent") is not None or v.get("data")
    }
    try:
        with open(USER_SESSIONS_PATH, "w") as f:
            json.dump(sessions_to_save, f, indent=4)
    except Exception as e:
        print(f"❌ Error saving sessions: {e}")


load_sessions()


def get_intent_response(message):
    message = message.lower()
    for intent in intents.get("intents", []):
        for pattern in intent.get("patterns", []):
            pattern_lower = pattern.lower()

            if re.search(r"\b" + re.escape(pattern_lower) + r"\b", message):
                return intent["tag"], random.choice(intent.get("responses", []))

            if len(pattern.split()) > 1 and pattern.lower() in message:
                return intent["tag"], random.choice(intent.get("responses", []))

    return None, None


def get_intent_response_by_tag(tag):
    for intent in intents.get("intents", []):
        if intent["tag"] == tag and intent.get("responses"):
            return random.choice(intent["responses"])

    for intent in intents.get("intents", []):
        if intent["tag"] == "fallback_generic":
            return random.choice(intent.get("responses", []))

    return "There's something off with the server. Reach out devs."


def parse_int(value):
    try:
        return int(value)
    except:
        return None


def parse_float(value):
    try:
        return float(value)
    except:
        return None


# --- Hydration Tip Generation (Original Logic) ---
def hydration_tip(
    activity_level_int,
    intensity_score,
    temperature,
    complication,
    is_indoors,
    is_windy_or_fanned,
    is_direct_sun,
    predicted_intake,
):

    tip_parts = []
    activity_display_str = activity_map_reverse.get(
        activity_level_int, "N/A"
    ).capitalize()

    HIGH_THRESHOLD = 3600
    LOW_THRESHOLD = 3000

    if intensity_score >= 0.6:
        tip_parts.append(
            f"💪 High Activity (Score: {intensity_score:.2f}): Your **{activity_display_str}** activity is strenuous. Sip water every 15–20 minutes during exercise. Consider electrolyte replacement if you sweat heavily."
        )
    elif intensity_score >= 0.3:
        tip_parts.append(
            f"🏃 Moderate Activity (Score: {intensity_score:.2f}): Your **{activity_display_str}** activity requires pre-hydration and post-hydration. Drink water evenly throughout the day, especially before and after workouts."
        )
    else:
        tip_parts.append(
            f"🧘 Low Activity (Score: {intensity_score:.2f}): Your **{activity_display_str}** activity is light. Even for light activity, maintain regular hydration to maintain focus."
        )

    if temperature > 30 and is_direct_sun == 1:
        tip_parts.append(
            f"🌞 **HIGH RISK:** You are in direct sun at {temperature}°C. Drink frequently and prioritize cool water."
        )
    elif temperature > 30:
        tip_parts.append(
            f"🌡️ Hot conditions detected ({temperature}°C), drink more frequently."
        )
    elif is_windy_or_fanned == 1:
        tip_parts.append(
            "💨 Air movement (fan/wind) accelerates water loss through evaporation. Keep a bottle handy and sip often."
        )
    elif is_indoors == 1 and temperature < 20:
        tip_parts.append(
            "🏠 Indoors and cool: You might not feel thirsty, but stay hydrated regularly to maintain focus."
        )

    if predicted_intake > HIGH_THRESHOLD:
        tip_parts.append(
            f"💧 Your recommended intake (~{predicted_intake:.0f} ml) is significantly higher. Consistent sipping is key—don't try to chug it all at once."
        )
    elif predicted_intake < LOW_THRESHOLD:
        tip_parts.append(
            f"💧 Your recommended intake (~{predicted_intake:.0f} ml) is lower than the typical range, but maintain steady small amounts throughout the day."
        )
    else:
        tip_parts.append(
            f"💧 Your recommended intake (~{predicted_intake:.0f} ml) is right in the typical range. Maintain steady sipping habits throughout the day."
        )

    if complication == 2:
        tip_parts.append(
            "⚠️ **HEALTH WARNING:** Severe health condition detected. Follow your doctor's exact guidance on fluid intake."
        )

    return "\n\n".join(tip_parts)


# --- Activity Mapping (Original Logic) ---
def map_activity_level_to_details(
    activity_level_int, sub_activity_name, age, weight, gender
):
    """Map activity level + user-chosen sub-activity to details used for prediction."""
    activity_details_map = {
        0: [  # Low activity
            {
                "activity_type": 4,
                "name": "Yoga/Stretching",
                "base_duration": 30,
                "base_pace": 0.0,
                "base_sweat": 1,
                "terrain": 0,
            },
            {
                "activity_type": 1,
                "name": "Light Running",
                "base_duration": 20,
                "base_pace": 5.0,
                "base_sweat": 2,
                "terrain": 0,
            },
            {
                "activity_type": 2,
                "name": "Easy Cycling",
                "base_duration": 25,
                "base_pace": 8.0,
                "base_sweat": 1,
                "terrain": 0,
            },
        ],
        1: [  # Medium activity
            {
                "activity_type": 3,
                "name": "Gym Workout",
                "base_duration": 60,
                "base_pace": 6.0,
                "base_sweat": 2,
                "terrain": 0,
            },
            {
                "activity_type": 1,
                "name": "Moderate Running",
                "base_duration": 45,
                "base_pace": 7.0,
                "base_sweat": 2,
                "terrain": 0,
            },
        ],
        2: [  # High activity
            {
                "activity_type": 1,
                "name": "Intense Running",
                "base_duration": 90,
                "base_pace": 9.5,
                "base_sweat": 3,
                "terrain": 1,
            },
            {
                "activity_type": 5,
                "name": "Intense Sports",
                "base_duration": 80,
                "base_pace": 8.5,
                "base_sweat": 3,
                "terrain": 1,
            },
        ],
    }

    sub_activity = next(
        (
            a
            for a in activity_details_map.get(activity_level_int, [])
            if a["name"].lower() == sub_activity_name.lower()
        ),
        activity_details_map.get(activity_level_int, [])[0],
    )

    duration = sub_activity["base_duration"]
    pace = sub_activity["base_pace"]
    sweat = sub_activity["base_sweat"]
    activity_type = sub_activity["activity_type"]
    terrain = sub_activity["terrain"]

    if age >= 55 and activity_level_int >= 1:
        duration -= 15
        pace = max(0.0, pace - 1.0)
        sweat = max(1, sweat - 1)
    elif age <= 25 and activity_level_int == 2:
        duration += 10
        pace += 0.5

    if weight >= 90.0 and activity_level_int >= 1:
        sweat = min(3, sweat + 1)
        pace = max(0.0, pace - 0.5)

    if gender == 1 and activity_level_int >= 1:
        sweat = min(3, sweat + 1)

    duration = max(10.0, duration)
    pace = max(0.0, pace)
    sweat = max(1, min(3, sweat))

    return {
        "activity_type": activity_type,
        "duration_minutes": duration,
        "pace": pace,
        "terrain_type": terrain,
        "sweat_level": sweat,
    }


def calculate_intensity_score(
    activity_type, duration_minutes, pace, terrain_type, sweat_level
):
    """Calculate intensity score based on activity type, duration, pace, and sweat level."""
    type_multiplier = {0: 0.1, 1: 1.0, 2: 0.8, 3: 0.6, 4: 0.3, 5: 1.2}.get(
        activity_type, 0.5
    )

    duration_factor = min(1.5, duration_minutes / 60.0)
    pace_factor = min(1.2, pace / 8.0) if pace > 0 else 1.0
    terrain_factor = 1.0 + (terrain_type * 0.15)
    sweat_factor = 1.0 + (sweat_level * 0.2)

    intentsity_score = (
        type_multiplier * duration_factor * pace_factor * terrain_factor * sweat_factor
    )
    return round(intentsity_score / 1.5, 2)


STANDARD_GLASS_ML = 250  # Standard size of a cup/glass in ml (approx 8 oz)

def get_gemma_response(user_message, chat_history):
    global ollama_client
    
    # In your app.py system prompt
    system_prompt = (
        "You are a professional assistant. Use Markdown for all responses: "
        "- Use '---' on a new line to separate numbering or distinct topics/sections or key differences or concepts."
        "- Use bold (**text**) for emphasis. "
        "- Use bulleted lists (- item) for steps or data. "
        "Maintain a clear, spaced-out structure."
    )

    messages_payload = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": user_message}]

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

    except Exception as e:
        print(f"Error generating Research response: {e}")
        return "I'm sorry, I'm having trouble reaching my research tools right now."


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    session_id = data.get("session_id", "default_user")
    local_storage_data = data.get("user_data", {})

    predicted_intake = 0.0

    if session_id not in user_sessions:
        user_sessions[session_id] = {
            "last_intent": None,
            "current_field": None,
            "data": {},
        }

    session = user_sessions[session_id]
    response_payload = {"response": "", "ask_for": None}

    # Pre-check if frontend sent all required data
    core_features = set(
        [
            "age",
            "gender",
            "weight",
            "activity",
            "complication",
            "humidity_scale", # <--- CRITICAL: CHECKING FOR SCALE
            "temperature",
        ]
    )
    is_data_complete_from_frontend = core_features.issubset(
        set(local_storage_data.keys())
    )

    if is_data_complete_from_frontend and not session.get("data"):
        for key, value in local_storage_data.items():
            session["data"][key] = value.lower() if isinstance(value, str) else value

        session["last_intent"] = "data_collection_complete"
        response_payload["response"] = (
            "Analyzing your profile data to provide a personalized hydration recommendation..."
        )
        detected_tag = None
        generic_response = None
    else:
        detected_tag, generic_response = get_intent_response(user_message)

    if "chat_history" not in session:
        session["chat_history"] = []

    # --- CORE LOGIC: Check Hydration Flow, otherwise Delegate to Ollama ---
    is_in_hydration_flow = session.get("last_intent") in [
        "ask_permission",
        "data_collection_started",
    ]
    is_hydration_intent = detected_tag == "start_data_collection"

    # IF we are chatting with the LLM (not hydration flow)
    if not is_in_hydration_flow and not is_hydration_intent:
        
        # 1. Pass the HISTORY to the function
        gemma_response_text = get_gemma_response(user_message, session["chat_history"])
        
        response_payload["response"] = gemma_response_text

        # 2. Update the History in Memory
        # Append User Message
        session["chat_history"].append({"role": "user", "content": user_message})
        # Append AI Response
        session["chat_history"].append({"role": "assistant", "content": gemma_response_text})
        
        # Keep history manageable (optional: keep last 10 turns to save memory)
        if len(session["chat_history"]) > 20:
            session["chat_history"] = session["chat_history"][-20:]

        response_payload["ask_for"] = None
        session["current_field"] = None

        save_sessions()
        return jsonify(response_payload)
    # --------------------------------------------------------------------

    # ----------------------------
    # START DATA COLLECTION
    # ----------------------------
    elif detected_tag == "start_data_collection":
        session["last_intent"] = "ask_permission"
        session["data"] = {}
        session["current_field"] = None
        time.sleep(1)
        response_payload["response"] = get_intent_response_by_tag("ask_permission")
        response_payload["ask_for"] = "permission_check"

    # ----------------------------
    # PERMISSION CHECK
    # ----------------------------
    elif session.get("last_intent") == "ask_permission":
        msg_lower = user_message.lower()
        time.sleep(1)
        if any(word in msg_lower for word in ["yes", "yup", "sure", "ok"]):
            for key, value in local_storage_data.items():
                session["data"][key] = (
                    value.lower() if isinstance(value, str) else value
                )

            next_field = get_first_missing_feature(session["data"], local_storage_data)
            confirmation_msg = get_intent_response_by_tag("confirmation")

            if next_field is None:
                session["last_intent"] = "data_collection_complete"
                session["current_field"] = None
                response_payload["response"] = (
                    f"{confirmation_msg} Moving to prediction..."
                )
            else:
                session["last_intent"] = "data_collection_started"
                session["current_field"] = next_field
                response_payload["response"] = (
                    f"{confirmation_msg} {get_intent_response_by_tag(f'ask_{next_field}')}"
                )
                response_payload["ask_for"] = next_field

        elif any(word in msg_lower for word in ["no", "nope", "not now"]):
            session["last_intent"] = None
            session["current_field"] = None
            time.sleep(1)
            response_payload["response"] = get_intent_response_by_tag("denial")
        else:
            time.sleep(1)
            response_payload["response"] = get_intent_response_by_tag(
                "fallback_permission_retry"
            )
            response_payload["ask_for"] = "permission_check"

    # ----------------------------
    # DATA COLLECTION PHASE
    # ----------------------------
    elif session.get("last_intent") == "data_collection_started":
        current_field = session.get("current_field")
        input_value = user_message.strip()

        # Validate numeric fields (Note: "humidity" is replaced by "humidity_scale" here)
        if current_field in ["age", "weight", "temperature", "humidity_scale"]:
            if parse_numeric_text(input_value) is None:
                time.sleep(1)
                response_payload["response"] = (
                    f"Sorry, I need a valid number for {current_field}. Please try again."
                )
                response_payload["ask_for"] = current_field
                save_sessions()
                return jsonify(response_payload)
            # CRITICAL VALIDATION FOR SCALE: Ensure it is 1-5
            if current_field == "humidity_scale":
                scale_val = parse_int(input_value)
                if scale_val is None or not (1 <= scale_val <= 5):
                    time.sleep(1)
                    response_payload["response"] = (
                        "The humidity scale must be a number between 1 (very high) and 5 (very low). Please enter a valid scale value."
                    )
                    response_payload["ask_for"] = current_field
                    save_sessions()
                    return jsonify(response_payload)


        # Save input
        if current_field:
            session["data"][current_field] = (
                input_value.lower() if isinstance(input_value, str) else input_value
            )

        # Special handling: activity -> ask for sub-activity based on level
        if current_field == "activity":
            activity_level_int = activity_map.get(input_value.lower(), 0)
            session["data"][
                "activity_level_int"
            ] = activity_level_int  # store for later use

            sub_activity_options = {
                0: ["Yoga/Stretching", "Light Running", "Easy Cycling"],
                1: ["Gym Workout", "Moderate Running"],
                2: ["Intense Running", "Intense Sports"],
            }

            options_text = ", ".join(sub_activity_options.get(activity_level_int, []))
            session["current_field"] = "sub_activity"
            response_payload["response"] = (
                f"You chose {input_value.capitalize()} activity. "
                f"Which type of activity do you usually do? (Options: {options_text})"
            )
            response_payload["ask_for"] = "sub_activity"
            save_sessions()
            return jsonify(response_payload)

        # After sub-activity
        if current_field == "sub_activity":
            session["data"]["sub_activity"] = input_value
            next_field = get_first_missing_feature(session["data"], local_storage_data)
            session["current_field"] = next_field

            if next_field:
                response_payload["response"] = get_intent_response_by_tag(
                    f"ask_{next_field}"
                )
                response_payload["ask_for"] = next_field
            else:
                session["last_intent"] = "data_collection_complete"
                session["current_field"] = None
                response_payload["response"] = (
                    "Thank you! I have all the data. Calculating recommendation..."
                )
                response_payload["ask_for"] = None

            save_sessions()
            return jsonify(response_payload)

        # Normal flow for other fields
        next_field = get_first_missing_feature(session["data"], local_storage_data)
        if next_field:
            session["current_field"] = next_field
            response_payload["response"] = get_intent_response_by_tag(
                f"ask_{next_field}"
            )
            response_payload["ask_for"] = next_field
        else:
            session["last_intent"] = "data_collection_complete"
            session["current_field"] = None
            response_payload["response"] = (
                "Thank you! I have all the data. Calculating recommendation..."
            )
            response_payload["ask_for"] = None

    # ----------------------------
    # PREDICTION PHASE
    # ----------------------------
    if session.get("last_intent") == "data_collection_complete":
        d = session["data"]

        # Parse and validate data
        age = parse_int(d.get("age")) or default_values["age"]
        weight = parse_float(d.get("weight")) or default_values["weight"]
        gender = gender_map.get(d.get("gender", "").lower(), default_values["gender"])
        
        # --- CRITICAL CHANGE: PARSING HUMIDITY SCALE DIRECTLY ---
        # The scale (1-5) is now directly received from the user
        humidity_scale = parse_int(d.get("humidity_scale")) or 3 
        # (Note: raw_humidity_input and its calculation logic are removed)
        # --------------------------------------------------------

        temperature = (
            parse_numeric_text(d.get("temperature")) or default_values["temperature"]
        )
        activity_level = activity_map.get(
            d.get("activity", "").lower(), default_values["activity"]
        )
        sub_activity_name = d.get("sub_activity", "Yoga/Stretching")

        detailed_activity = map_activity_level_to_details(
            activity_level, sub_activity_name, age, weight, gender
        )

        activity_type = detailed_activity["activity_type"]
        duration_minutes = detailed_activity["duration_minutes"]
        pace = detailed_activity["pace"]
        terrain_type = detailed_activity["terrain_type"]
        sweat_level = detailed_activity["sweat_level"]

        intensity_score = calculate_intensity_score(
            activity_type, duration_minutes, pace, terrain_type, sweat_level
        )

        complication_str_raw = d.get("complication", "").lower()
        if complication_str_raw in complication_map:
            complication = complication_map[complication_str_raw]
        elif any(
            word in complication_str_raw
            for word in ["diabetes", "renal", "kidney", "heart"]
        ):
            complication = 2
        else:
            complication = 0

        is_indoors = indoors_map.get(
            d.get("is_indoors", "").lower(), default_values["is_indoors"]
        )
        is_ground_wet = wet_ground_map.get(
            d.get("is_ground_wet", "").lower(), default_values["is_ground_wet"]
        )
        is_windy_or_fanned = binary_map.get(
            d.get("is_windy_or_fanned", "").lower(),
            default_values["is_windy_or_fanned"],
        )
        is_direct_sun = binary_map.get(
            d.get("is_direct_sun", "").lower(), default_values["is_direct_sun"]
        )

        # Model prediction
        if model and scaler:
            # --- CRITICAL CHANGE: 16-FEATURE INPUT ARRAY (REMOVED raw_humidity_input) ---
            # NOTE: The order MUST match the V8 model training!
            X = np.array(
                [
                    [
                        age,
                        gender,
                        weight,
                        humidity_scale, # <--- Feature #4 (The new input)
                        temperature,
                        complication,
                        is_indoors,
                        is_ground_wet,
                        is_windy_or_fanned,
                        is_direct_sun,
                        activity_type,
                        duration_minutes,
                        pace,
                        terrain_type,
                        sweat_level,
                        intensity_score,
                    ]
                ]
            )
            # --------------------------------------------------------------------------
            X_scaled = scaler.transform(X)
            time.sleep(2)
            predicted_intake = float(model.predict(X_scaled)[0][0])
            num_glasses = predicted_intake / STANDARD_GLASS_ML

            tip_text = hydration_tip(
                activity_level_int=activity_level,
                intensity_score=intensity_score,
                temperature=temperature,
                complication=complication,
                is_indoors=is_indoors,
                is_windy_or_fanned=is_windy_or_fanned,
                is_direct_sun=is_direct_sun,
                predicted_intake=predicted_intake,
            )

            environment_text = f"{'Indoors' if is_indoors else 'Outdoors'}, Ground {'Wet' if is_ground_wet else 'Dry'}"
            if is_windy_or_fanned:
                environment_text += ", Strong Wind/Fan"
            if is_direct_sun:
                environment_text += ", Direct Sun ☀️"

            summary_obj = {
                "title": "🧾 Hydration Summary",
                "description": "Using the collected additional information, our model predicted or calculated how much water you should to take.",
                "bullets": [
                    {
                        "indent": 1,
                        "text": f"👤 Profile: {age} yo, {gender_map_reverse.get(gender, 'N/A').capitalize()}, {weight:.1f} kg",
                    },
                    {
                        "indent": 1,
                        "text": f"🏃 Activity: {activity_map_reverse.get(activity_level, 'N/A').capitalize()} - {sub_activity_name} (Score: {intensity_score:.2f})",
                    },
                    {
                        "indent": 1,
                        "text": f"⏱️ Estimated Duration/Pace: {duration_minutes:.0f} min, {pace:.1f} km/h",
                    },
                    {
                        "indent": 1,
                        # IMPORTANT: Raw humidity is no longer available; display only scale
                        "text": f"🌡️ Conditions: {temperature:.1f}°C, Humidity Scale: {humidity_scale}",
                    },
                    {"indent": 1, "text": f"🏠 Environment: {environment_text}"},
                    {
                        "indent": 1,
                        "text": f"🩺 Complication: {complication_map_reverse.get(complication, 'N/A').capitalize()}",
                    },
                ],
                "caution_text": "The predicted water intake is not always accurate, and this is not alternative to any health expert. Consider the result as a guide.",
                "recommended_intake": f"~{predicted_intake:.0f} ml or {num_glasses:.1f} glasses",
                "tip": tip_text,
            }

            session["last_intent"] = None
            session["current_field"] = None
            response_payload["response"] = get_intent_response_by_tag(
                "response_loading"
            )
            response_payload["summary"] = summary_obj
            response_payload["ask_for"] = None
        else:
            session["last_intent"] = None
            session["current_field"] = None
            response_payload["response"] = get_intent_response_by_tag(
                "fallback_model_error"
            )
            response_payload["ask_for"] = None
    
    save_sessions()
    return jsonify(response_payload)


# ==================================
# DEDICATED PREDICTION ENDPOINT 
# ==================================
@app.route("/ai-api/predict-goal", methods=["POST"])
def predict_hydration_goal():
    """
    Predicts the daily hydration goal (in Liters) based on user profile data.
    Now synchronized with /chat logic to include environmental and specific activity factors.
    """
    try:
        data = request.get_json(silent=True) or {}

        # --- 1. Parse Inputs (Profile) ---
        age = parse_int(data.get("age")) or default_values["age"]
        weight = parse_float(data.get("weight")) or default_values["weight"]
        gender = gender_map.get(data.get("gender", "").lower(), default_values["gender"])
        activity_level = activity_map.get(data.get("activity", "").lower(), default_values["activity"])
        
        # --- 2. Parse Inputs (Environment) ---
        # RAW Humidity Input
        humidity_scale = parse_int(data.get("humidity_scale")) or default_values["humidity_scale"]

        temperature = parse_numeric_text(data.get("temperature")) or default_values["temperature"]
        
        is_indoors = indoors_map.get(
            data.get("is_indoors", "").lower(), default_values["is_indoors"]
        )
        is_ground_wet = wet_ground_map.get(
            data.get("is_ground_wet", "").lower(), default_values["is_ground_wet"]
        )
        is_windy_or_fanned = binary_map.get(
            data.get("is_windy_or_fanned", "").lower(), default_values["is_windy_or_fanned"],
        )
        is_direct_sun = binary_map.get(
            data.get("is_direct_sun", "").lower(), default_values["is_direct_sun"]
        )

        # --- 3. Parse Complications ---
        complication_str_raw = data.get("complication", "").lower()
        if complication_str_raw in complication_map:
            complication = complication_map[complication_str_raw]
        else:
            if any(word in complication_str_raw for word in ["diabetes", "renal", "kidney", "heart", "severe"]):
                complication = 2
            elif any(word in complication_str_raw for word in ["fever", "heavy_exercise", "mild"]):
                complication = 1
            else:
                complication = 0

        # --- 4. Activity Mapping ---
        # Try to get specific sub_activity if provided, otherwise default to first in list
        activity_details_map = {
            0: [  # Low activity
                {"activity_type": 4, "name": "Yoga/Stretching", "base_duration": 30, "base_pace": 0.0, "base_sweat": 1, "terrain": 0},
                {"activity_type": 1, "name": "Light Running", "base_duration": 20, "base_pace": 5.0, "base_sweat": 2, "terrain": 0},
                {"activity_type": 2, "name": "Easy Cycling", "base_duration": 25, "base_pace": 8.0, "base_sweat": 1, "terrain": 0},
            ],
            1: [  # Medium activity
                {"activity_type": 3, "name": "Gym Workout", "base_duration": 60, "base_pace": 6.0, "base_sweat": 2, "terrain": 0},
                {"activity_type": 1, "name": "Moderate Running", "base_duration": 45, "base_pace": 7.0, "base_sweat": 2, "terrain": 0},
            ],
            2: [  # High activity
                {"activity_type": 1, "name": "Intense Running", "base_duration": 90, "base_pace": 9.5, "base_sweat": 3, "terrain": 1},
                {"activity_type": 5, "name": "Intense Sports", "base_duration": 80, "base_pace": 8.5, "base_sweat": 3, "terrain": 1},
            ],
        }

        # Check if sub_activity is passed, otherwise pick default based on activity level
        input_sub_activity = data.get("sub_activity")
        if input_sub_activity:
            sub_activity_name = input_sub_activity
        else:
            sub_activity_name = activity_details_map.get(activity_level, [])[0]["name"]

        detailed_activity = map_activity_level_to_details(
            activity_level, sub_activity_name, age, weight, gender
        )

        activity_type = detailed_activity["activity_type"]
        duration_minutes = detailed_activity["duration_minutes"]
        pace = detailed_activity["pace"]
        terrain_type = detailed_activity["terrain_type"]
        sweat_level = detailed_activity["sweat_level"]

        # Calculate intensity score
        intensity_score = calculate_intensity_score(
            activity_type, duration_minutes, pace, terrain_type, sweat_level
        )

        # --- 5. Model Prediction ---
        if model and scaler:
            X = np.array([[
                age, 
                gender, 
                weight,
                humidity_scale,
                temperature, 
                complication, 
                is_indoors, 
                is_ground_wet, 
                is_windy_or_fanned, 
                is_direct_sun, 
                activity_type, 
                duration_minutes,
                pace, 
                terrain_type, 
                sweat_level, 
                intensity_score
            ]])
            
            X_scaled = scaler.transform(X)
            predicted_intake_ml = float(model.predict(X_scaled)[0][0])

            # Generate message
            if complication == 2:
                message = "Goal calculated with severe health caution. Consult a doctor."
            elif intensity_score >= 0.6:
                message = "Goal adjusted for your high activity and biometrics."
            else:
                message = "Your personalized goal has been calculated successfully."

            return jsonify({
                "status": "success",
                "predicted_goal_ml": predicted_intake_ml,
                "predicted_message": message
            })

        else:
            return jsonify({
                "status": "error",
                "predicted_goal_liters": 2.5,
                "predicted_message": "Prediction model is unavailable. Default goal set."
            }), 500

    except Exception as e:
        print(f"Error in dedicated prediction endpoint: {e}")
        return jsonify({
            "status": "error",
            "predicted_goal_liters": 2.5,
            "predicted_message": "An internal server error occurred during prediction."
        }), 500

# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
