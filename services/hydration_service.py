import re
import json
import random
import time
import numpy as np
import tensorflow as tf
from tf_keras import losses
from tf_keras import metrics
import joblib

from config import (
    MODEL_PATH, INTENTS_PATH, SCALER_PATH, REQUIRED_FEATURES,
    GENDER_MAP, ACTIVITY_MAP, COMPLICATION_MAP, INDOORS_MAP,
    WET_GROUND_MAP, BINARY_MAP, GENDER_MAP_REVERSE,
    ACTIVITY_MAP_REVERSE, COMPLICATION_MAP_REVERSE, STANDARD_GLASS_ML,
    DEFAULT_VALUES
)

class HydrationService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.intents = {"intents": []}
        self.load_assets()

    def load_assets(self):
        try:
            custom_objects = {
                "mse": losses.MeanSquaredError(),
                "mae": metrics.MeanAbsoluteError(),
            }
            self.model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
            self.scaler = joblib.load(SCALER_PATH)
            with open(INTENTS_PATH, "r", encoding="utf-8") as f:
                self.intents = json.load(f)
            print("✅ Model, scaler, and intents loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading assets: {e}")
            self.model = None
            self.intents = {"intents": []}

    def parse_numeric_text(self, value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        match = re.search(r"[-+]?\d*\.?\d+", str(value))
        if match:
            return float(match.group())
        return None

    def parse_int(self, value):
        try:
            return int(value)
        except:
            return None

    def parse_float(self, value):
        try:
            return float(value)
        except:
            return None

    def get_first_missing_feature(self, session_data, local_data):
        all_data = {
            k: v.lower() if isinstance(v, str) else v for k, v in local_data.items()
        }
        all_data.update(session_data)

        for feature in REQUIRED_FEATURES:
            value = all_data.get(feature)
            if value is None or (isinstance(value, str) and value.strip() == ""):
                return feature
            if (
                feature in ["age", "weight", "humidity", "temperature"]
                and self.parse_numeric_text(value) is None
            ):
                return feature
            if feature == "gender" and str(value).lower() not in GENDER_MAP:
                return feature
            if feature == "activity" and str(value).lower() not in ACTIVITY_MAP:
                return feature
            if feature == "is_indoors" and str(value).lower() not in INDOORS_MAP:
                return feature
            if feature == "is_ground_wet" and str(value).lower() not in WET_GROUND_MAP:
                return feature
            if (
                feature in ["is_windy_or_fanned", "is_direct_sun"]
                and str(value).lower() not in BINARY_MAP
            ):
                return feature
        return None

    def get_intent_response(self, message):
        message = message.lower()
        for intent in self.intents.get("intents", []):
            for pattern in intent.get("patterns", []):
                pattern_lower = pattern.lower()
                if re.search(r"\b" + re.escape(pattern_lower) + r"\b", message):
                    return intent["tag"], random.choice(intent.get("responses", []))
                if len(pattern.split()) > 1 and pattern.lower() in message:
                    return intent["tag"], random.choice(intent.get("responses", []))
        return None, None

    def get_intent_response_by_tag(self, tag):
        for intent in self.intents.get("intents", []):
            if intent["tag"] == tag and intent.get("responses"):
                return random.choice(intent["responses"])
        for intent in self.intents.get("intents", []):
            if intent["tag"] == "fallback_generic":
                return random.choice(intent.get("responses", []))
        return "There's something off with the server. Reach out devs."

    def hydration_tip(self, activity_level_int, intensity_score, temperature, complication, is_indoors, is_windy_or_fanned, is_direct_sun, predicted_intake):
        tip_parts = []
        activity_display_str = ACTIVITY_MAP_REVERSE.get(activity_level_int, "N/A").capitalize()
        HIGH_THRESHOLD = 3600
        LOW_THRESHOLD = 3000

        if intensity_score >= 0.6:
            tip_parts.append(f"💪 High Activity (Score: {intensity_score:.2f}): Your **{activity_display_str}** activity is strenuous. Sip water every 15–20 minutes during exercise. Consider electrolyte replacement if you sweat heavily.")
        elif intensity_score >= 0.3:
            tip_parts.append(f"🏃 Moderate Activity (Score: {intensity_score:.2f}): Your **{activity_display_str}** activity requires pre-hydration and post-hydration. Drink water evenly throughout the day, especially before and after workouts.")
        else:
            tip_parts.append(f"🧘 Low Activity (Score: {intensity_score:.2f}): Your **{activity_display_str}** activity is light. Even for light activity, maintain regular hydration to maintain focus.")

        if temperature > 30 and is_direct_sun == 1:
            tip_parts.append(f"🌞 **HIGH RISK:** You are in direct sun at {temperature}°C. Drink frequently and prioritize cool water.")
        elif temperature > 30:
            tip_parts.append(f"🌡️ Hot conditions detected ({temperature}°C), drink more frequently.")
        elif is_windy_or_fanned == 1:
            tip_parts.append("💨 Air movement (fan/wind) accelerates water loss through evaporation. Keep a bottle handy and sip often.")
        elif is_indoors == 1 and temperature < 20:
            tip_parts.append("🏠 Indoors and cool: You might not feel thirsty, but stay hydrated regularly to maintain focus.")

        if predicted_intake > HIGH_THRESHOLD:
            tip_parts.append(f"💧 Your recommended intake (~{predicted_intake:.0f} ml) is significantly higher. Consistent sipping is key—don't try to chug it all at once.")
        elif predicted_intake < LOW_THRESHOLD:
            tip_parts.append(f"💧 Your recommended intake (~{predicted_intake:.0f} ml) is lower than the typical range, but maintain steady small amounts throughout the day.")
        else:
            tip_parts.append(f"💧 Your recommended intake (~{predicted_intake:.0f} ml) is right in the typical range. Maintain steady sipping habits throughout the day.")

        if complication == 2:
            tip_parts.append("⚠️ **HEALTH WARNING:** Severe health condition detected. Follow your doctor's exact guidance on fluid intake.")

        return "\n\n".join(tip_parts)

    def map_activity_level_to_details(self, activity_level_int, sub_activity_name, age, weight, gender):
        activity_details_map = {
            0: [
                {"activity_type": 4, "name": "Yoga/Stretching", "base_duration": 30, "base_pace": 0.0, "base_sweat": 1, "terrain": 0},
                {"activity_type": 1, "name": "Light Running", "base_duration": 20, "base_pace": 5.0, "base_sweat": 2, "terrain": 0},
                {"activity_type": 2, "name": "Easy Cycling", "base_duration": 25, "base_pace": 8.0, "base_sweat": 1, "terrain": 0},
            ],
            1: [
                {"activity_type": 3, "name": "Gym Workout", "base_duration": 60, "base_pace": 6.0, "base_sweat": 2, "terrain": 0},
                {"activity_type": 1, "name": "Moderate Running", "base_duration": 45, "base_pace": 7.0, "base_sweat": 2, "terrain": 0},
            ],
            2: [
                {"activity_type": 1, "name": "Intense Running", "base_duration": 90, "base_pace": 9.5, "base_sweat": 3, "terrain": 1},
                {"activity_type": 5, "name": "Intense Sports", "base_duration": 80, "base_pace": 8.5, "base_sweat": 3, "terrain": 1},
            ],
        }

        sub_activity = next(
            (a for a in activity_details_map.get(activity_level_int, []) if a["name"].lower() == sub_activity_name.lower()),
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

    def calculate_intensity_score(self, activity_type, duration_minutes, pace, terrain_type, sweat_level):
        type_multiplier = {0: 0.1, 1: 1.0, 2: 0.8, 3: 0.6, 4: 0.3, 5: 1.2}.get(activity_type, 0.5)
        duration_factor = min(1.5, duration_minutes / 60.0)
        pace_factor = min(1.2, pace / 8.0) if pace > 0 else 1.0
        terrain_factor = 1.0 + (terrain_type * 0.15)
        sweat_factor = 1.0 + (sweat_level * 0.2)
        intentsity_score = (type_multiplier * duration_factor * pace_factor * terrain_factor * sweat_factor)
        return round(intentsity_score / 1.5, 2)

    def predict_intake(self, data):
        age = self.parse_int(data.get("age")) or DEFAULT_VALUES["age"]
        weight = self.parse_float(data.get("weight")) or DEFAULT_VALUES["weight"]
        gender = GENDER_MAP.get(data.get("gender", "").lower(), DEFAULT_VALUES["gender"])
        humidity_scale = self.parse_int(data.get("humidity_scale")) or DEFAULT_VALUES["humidity_scale"]
        temperature = self.parse_numeric_text(data.get("temperature")) or DEFAULT_VALUES["temperature"]
        activity_level = ACTIVITY_MAP.get(data.get("activity", "").lower(), DEFAULT_VALUES["activity"])
        sub_activity_name = data.get("sub_activity", "Yoga/Stretching")
        
        complication_str_raw = data.get("complication", "").lower()
        if complication_str_raw in COMPLICATION_MAP:
            complication = COMPLICATION_MAP[complication_str_raw]
        elif any(word in complication_str_raw for word in ["diabetes", "renal", "kidney", "heart"]):
            complication = 2
        else:
            complication = 0

        is_indoors = INDOORS_MAP.get(data.get("is_indoors", "").lower(), DEFAULT_VALUES["is_indoors"])
        is_ground_wet = WET_GROUND_MAP.get(data.get("is_ground_wet", "").lower(), DEFAULT_VALUES["is_ground_wet"])
        is_windy_or_fanned = BINARY_MAP.get(data.get("is_windy_or_fanned", "").lower(), DEFAULT_VALUES["is_windy_or_fanned"])
        is_direct_sun = BINARY_MAP.get(data.get("is_direct_sun", "").lower(), DEFAULT_VALUES["is_direct_sun"])

        detailed_activity = self.map_activity_level_to_details(activity_level, sub_activity_name, age, weight, gender)
        
        # Unpack detailed activity
        activity_type = detailed_activity["activity_type"]
        duration_minutes = detailed_activity["duration_minutes"]
        pace = detailed_activity["pace"]
        terrain_type = detailed_activity["terrain_type"]
        sweat_level = detailed_activity["sweat_level"]
        
        intensity_score = self.calculate_intensity_score(activity_type, duration_minutes, pace, terrain_type, sweat_level)

        if self.model and self.scaler:
             X = np.array([[
                age, gender, weight, humidity_scale, temperature, complication,
                is_indoors, is_ground_wet, is_windy_or_fanned, is_direct_sun,
                activity_type, duration_minutes, pace, terrain_type, sweat_level, intensity_score
            ]])
             X_scaled = self.scaler.transform(X)
             predicted_intake = float(self.model.predict(X_scaled)[0][0])
        else:
            predicted_intake = 2500.0 # Default fallback

        return {
            "predicted_intake": predicted_intake,
            "intensity_score": intensity_score,
            "profile": {"age": age, "weight": weight, "gender": gender},
            "environment": {
                "temperature": temperature, 
                "humidity_scale": humidity_scale,
                "is_indoors": is_indoors,
                "is_ground_wet": is_ground_wet,
                "is_windy_or_fanned": is_windy_or_fanned,
                "is_direct_sun": is_direct_sun
            },
            "activity": {
                "level": activity_level,
                "name": sub_activity_name,
                "duration": duration_minutes,
                "pace": pace
            },
            "complication": complication
        }

hydration_service = HydrationService()
