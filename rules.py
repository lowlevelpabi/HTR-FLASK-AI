def get_health_tip(prompt):
    prompt = prompt.lower()

    if "drink" in prompt or "water" in prompt or "thirsty" in prompt:
        return "Remember to drink at least 8 glasses of water a day to stay hydrated!"
    elif "tired" in prompt or "weak" in prompt:
        return "You might dehydrated. Drink water and rest well!"
    elif "exercise" in prompt:
        return "After exercise, rehydrate with water to help your body recover."
    elif "hot" in prompt or "sweat" in prompt:
        return "Drink water every 30 minutes during hot weather."
    else:
        return "Stay hydraded and maintain a balanced water intake!"