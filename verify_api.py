import requests
import sys

def run_verification():
    print("🚀 Starting Verification (Client Mode)...")
    base_url = "http://192.168.1.60:5000"

    try:
        # Test 1: Chat Endpoint (AI Flow)
        print("\n🧪 Test 1: Chat Endpoint (AI Flow)")
        payload = {
            "message": "Hello, how are you?",
            "session_id": "test_user_verify",
            "user_data": {}
        }
        try:
            res = requests.post(f"{base_url}/chat", json=payload, timeout=30)
            if res.status_code == 200:
                print("✅ Chat AI Flow: Success")
                print("Response:", str(res.json().get("response"))[:100] + "...")
            else:
                print(f"❌ Chat AI Flow: Failed ({res.status_code})")
                print(res.text)
        except requests.exceptions.ConnectionError:
             print("❌ Connection Refused. Is the server running?")
             return

        # Test 2: Chat Endpoint (Hydration Flow - Start)
        print("\n🧪 Test 2: Chat Endpoint (Hydration Flow - Start)")
        payload = {
            "message": "I want to track hydration",
            "session_id": "test_user_verify",
            "user_data": {"age": "25", "weight": "70"}
        }
        res = requests.post(f"{base_url}/chat", json=payload, timeout=30)
        if res.status_code == 200:
            print("✅ Chat Hydration Flow: Success")
            print("Response:", res.json().get("response"))
        else:
            print(f"❌ Chat Hydration Flow: Failed ({res.status_code})")

        # Test 3: Dedicated Prediction Endpoint
        print("\n🧪 Test 3: Dedicated Prediction Endpoint")
        payload = {
            "age": 25,
            "weight": 70,
            "gender": "male",
            "activity": "medium",
            "temperature": 30,
            "humidity_scale": 3,
            "sub_activity": "Gym Workout"
        }
        res = requests.post(f"{base_url}/ai-api/predict-goal", json=payload, timeout=30)
        if res.status_code == 200:
             data = res.json()
             if data["status"] == "success":
                 print(f"✅ Prediction: Success (Goal: {data['predicted_goal_ml']} ml)")
             else:
                 print(f"❌ Prediction: Logical Error ({data['predicted_message']})")
        else:
            print(f"❌ Prediction: HTTP Error ({res.status_code})")
            print(res.text)

    except Exception as e:
        print(f"❌ Verification Failed: {e}")

if __name__ == "__main__":
    run_verification()
