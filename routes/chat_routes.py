import time
from flask import Blueprint, request, jsonify
from services.session_service import session_service
from services.ai_service import ai_service
from services.hydration_service import hydration_service
from config import (
    ACTIVITY_MAP, GENDER_MAP_REVERSE, ACTIVITY_MAP_REVERSE, 
    COMPLICATION_MAP_REVERSE, STANDARD_GLASS_ML, DEFAULT_VALUES,
    GENDER_MAP, COMPLICATION_MAP, INDOORS_MAP, WET_GROUND_MAP, BINARY_MAP
)

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    session_id = data.get("session_id", "default_user")
    local_storage_data = data.get("user_data", {})

    session = session_service.get_session(session_id)
    response_payload = {"response": "", "ask_for": None}

    # Pre-check if frontend sent all required data
    core_features = set(["age", "gender", "weight", "activity", "complication", "humidity_scale", "temperature"])
    is_data_complete_from_frontend = core_features.issubset(set(local_storage_data.keys()))

    detected_tag = None
    
    if is_data_complete_from_frontend and not session.get("data"):
         for key, value in local_storage_data.items():
             session["data"][key] = value.lower() if isinstance(value, str) else value
         session["last_intent"] = "data_collection_complete"
         response_payload["response"] = "Analyzing your profile data to provide a personalized hydration recommendation..."
    else:
         detected_tag, generic_response = hydration_service.get_intent_response(user_message)

    # --- CORE LOGIC: Check Hydration Flow, otherwise Delegate to Ollama ---
    is_in_hydration_flow = session.get("last_intent") in ["ask_permission", "data_collection_started"]
    is_hydration_intent = detected_tag == "start_data_collection"

    # IF we are chatting with the LLM (not hydration flow)
    if not is_in_hydration_flow and not is_hydration_intent:
        gemma_response_text = ai_service.get_gemma_response(user_message, session["chat_history"])
        response_payload["response"] = gemma_response_text
        
        session["chat_history"].append({"role": "user", "content": user_message})
        session["chat_history"].append({"role": "assistant", "content": gemma_response_text})
        
        if len(session["chat_history"]) > 20:
             session["chat_history"] = session["chat_history"][-20:]

        response_payload["ask_for"] = None
        session["current_field"] = None
        session_service.save_sessions()
        return jsonify(response_payload)

    # ----------------------------
    # START DATA COLLECTION
    # ----------------------------
    elif detected_tag == "start_data_collection":
        session["last_intent"] = "ask_permission"
        session["data"] = {}
        session["current_field"] = None
        time.sleep(1)
        response_payload["response"] = hydration_service.get_intent_response_by_tag("ask_permission")
        response_payload["ask_for"] = "permission_check"

    # ----------------------------
    # PERMISSION CHECK
    # ----------------------------
    elif session.get("last_intent") == "ask_permission":
        msg_lower = user_message.lower()
        time.sleep(1)
        if any(word in msg_lower for word in ["yes", "yup", "sure", "ok"]):
            for key, value in local_storage_data.items():
                session["data"][key] = value.lower() if isinstance(value, str) else value

            next_field = hydration_service.get_first_missing_feature(session["data"], local_storage_data)
            confirmation_msg = hydration_service.get_intent_response_by_tag("confirmation")

            if next_field is None:
                session["last_intent"] = "data_collection_complete"
                session["current_field"] = None
                response_payload["response"] = f"{confirmation_msg} Moving to prediction..."
            else:
                session["last_intent"] = "data_collection_started"
                session["current_field"] = next_field
                response_payload["response"] = f"{confirmation_msg} {hydration_service.get_intent_response_by_tag(f'ask_{next_field}')}"
                response_payload["ask_for"] = next_field
        
        elif any(word in msg_lower for word in ["no", "nope", "not now"]):
             session_service.clear_session(session_id)
             time.sleep(1)
             response_payload["response"] = hydration_service.get_intent_response_by_tag("denial")
        else:
             time.sleep(1)
             response_payload["response"] = hydration_service.get_intent_response_by_tag("fallback_permission_retry")
             response_payload["ask_for"] = "permission_check"

    # ----------------------------
    # DATA COLLECTION PHASE
    # ----------------------------
    elif session.get("last_intent") == "data_collection_started":
        current_field = session.get("current_field")
        input_value = user_message.strip()

        if current_field in ["age", "weight", "temperature", "humidity_scale"]:
             if hydration_service.parse_numeric_text(input_value) is None:
                 time.sleep(1)
                 response_payload["response"] = f"Sorry, I need a valid number for {current_field}. Please try again."
                 response_payload["ask_for"] = current_field
                 session_service.save_sessions()
                 return jsonify(response_payload)
             
             if current_field == "humidity_scale":
                 scale_val = hydration_service.parse_int(input_value)
                 if scale_val is None or not (1 <= scale_val <= 5):
                     time.sleep(1)
                     response_payload["response"] = "The humidity scale must be a number between 1 (very high) and 5 (very low). Please enter a valid scale value."
                     response_payload["ask_for"] = current_field
                     session_service.save_sessions()
                     return jsonify(response_payload)
        
        if current_field:
            session["data"][current_field] = input_value.lower() if isinstance(input_value, str) else input_value

        if current_field == "activity":
             activity_level_int = ACTIVITY_MAP.get(input_value.lower(), 0)
             session["data"]["activity_level_int"] = activity_level_int
             
             sub_activity_options = {
                0: ["Yoga/Stretching", "Light Running", "Easy Cycling"],
                1: ["Gym Workout", "Moderate Running"],
                2: ["Intense Running", "Intense Sports"],
             }
             options_text = ", ".join(sub_activity_options.get(activity_level_int, []))
             session["current_field"] = "sub_activity"
             response_payload["response"] = f"You chose {input_value.capitalize()} activity. Which type of activity do you usually do? (Options: {options_text})"
             response_payload["ask_for"] = "sub_activity"
             session_service.save_sessions()
             return jsonify(response_payload)
        
        if current_field == "sub_activity":
             session["data"]["sub_activity"] = input_value
             next_field = hydration_service.get_first_missing_feature(session["data"], local_storage_data)
             session["current_field"] = next_field

             if next_field:
                 response_payload["response"] = hydration_service.get_intent_response_by_tag(f"ask_{next_field}")
                 response_payload["ask_for"] = next_field
             else:
                 session["last_intent"] = "data_collection_complete"
                 session["current_field"] = None
                 response_payload["response"] = "Thank you! I have all the data. Calculating recommendation..."
                 response_payload["ask_for"] = None
             
             session_service.save_sessions()
             return jsonify(response_payload)
        
        next_field = hydration_service.get_first_missing_feature(session["data"], local_storage_data)
        if next_field:
             session["current_field"] = next_field
             response_payload["response"] = hydration_service.get_intent_response_by_tag(f"ask_{next_field}")
             response_payload["ask_for"] = next_field
        else:
             session["last_intent"] = "data_collection_complete"
             session["current_field"] = None
             response_payload["response"] = "Thank you! I have all the data. Calculating recommendation..."
             response_payload["ask_for"] = None

    # ----------------------------
    # PREDICTION PHASE
    # ----------------------------
    if session.get("last_intent") == "data_collection_complete":
        time.sleep(2)
        prediction_result = hydration_service.predict_intake(session["data"])
        predicted_intake = prediction_result["predicted_intake"]
        
        num_glasses = predicted_intake / STANDARD_GLASS_ML
        
        p = prediction_result
        tip_text = hydration_service.hydration_tip(
            activity_level_int=p["activity"]["level"],
            intensity_score=p["intensity_score"],
            temperature=p["environment"]["temperature"],
            complication=p["complication"],
            is_indoors=p["environment"]["is_indoors"],
            is_windy_or_fanned=p["environment"]["is_windy_or_fanned"],
            is_direct_sun=p["environment"]["is_direct_sun"],
            predicted_intake=predicted_intake
        )

        environment_text = f"{'Indoors' if p['environment']['is_indoors'] else 'Outdoors'}, Ground {'Wet' if p['environment']['is_ground_wet'] else 'Dry'}"
        if p["environment"]["is_windy_or_fanned"]:
            environment_text += ", Strong Wind/Fan"
        if p["environment"]["is_direct_sun"]:
            environment_text += ", Direct Sun ☀️"

        summary_obj = {
            "title": "🧾 Hydration Summary",
            "description": "Using the collected additional information, our model predicted or calculated how much water you should to take.",
            "bullets": [
                {"indent": 1, "text": f"👤 Profile: {p['profile']['age']} yo, {GENDER_MAP_REVERSE.get(p['profile']['gender'], 'N/A').capitalize()}, {p['profile']['weight']:.1f} kg"},
                {"indent": 1, "text": f"🏃 Activity: {ACTIVITY_MAP_REVERSE.get(p['activity']['level'], 'N/A').capitalize()} - {p['activity']['name']} (Score: {p['intensity_score']:.2f})"},
                {"indent": 1, "text": f"⏱️ Estimated Duration/Pace: {p['activity']['duration']:.0f} min, {p['activity']['pace']:.1f} km/h"},
                {"indent": 1, "text": f"🌡️ Conditions: {p['environment']['temperature']:.1f}°C, Humidity Scale: {p['environment']['humidity_scale']}"},
                {"indent": 1, "text": f"🏠 Environment: {environment_text}"},
                {"indent": 1, "text": f"🩺 Complication: {COMPLICATION_MAP_REVERSE.get(p['complication'], 'N/A').capitalize()}"},
            ],
            "caution_text": "The predicted water intake is not always accurate, and this is not alternative to any health expert. Consider the result as a guide.",
            "recommended_intake": f"~{predicted_intake:.0f} ml or {num_glasses:.1f} glasses",
            "tip": tip_text,
        }

        session_service.clear_session(session_id)
        response_payload["response"] = hydration_service.get_intent_response_by_tag("response_loading")
        response_payload["summary"] = summary_obj
        response_payload["ask_for"] = None

    session_service.save_sessions()
    return jsonify(response_payload)

@chat_bp.route("/ai-api/predict-goal", methods=["POST"])
def predict_hydration_goal_route():
    try:
        data = request.get_json(silent=True) or {}
        result = hydration_service.predict_intake(data)
        
        predicted_intake_ml = result["predicted_intake"]
        complication = result["complication"]
        intensity_score = result["intensity_score"]

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

    except Exception as e:
        print(f"Error in dedicated prediction endpoint: {e}")
        return jsonify({
            "status": "error",
            "predicted_goal_liters": 2.5,
            "predicted_message": "An internal server error occurred during prediction."
        }), 500
