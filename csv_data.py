import pandas as pd
import numpy as np
import datetime
import random
import string

np.random.seed(42)
data_size = 500

# ---------------------------
# 1. BASIC USER CHARACTERISTICS
# ---------------------------
age = np.random.randint(18, 58, size=data_size)
gender = np.random.randint(0, 2, size=data_size)
weight = np.random.randint(50, 100, size=data_size)

# ---------------------------
# 2. ENVIRONMENTAL FACTORS
# ---------------------------
# --- REMOVED RAW 'humidity' VARIABLE ---
# humidity = np.random.randint(0, 101, size=data_size)
raw_humidity_for_calc = np.random.randint(0, 101, size=data_size) # Temp variable for generating scale
temperature = np.random.randint(15, 40, size=data_size)
is_indoors = np.random.randint(0, 2, size=data_size)
is_ground_wet = np.random.randint(0, 2, size=data_size)
is_windy_or_fanned = np.random.randint(0, 2, size=data_size)
is_direct_sun = np.random.randint(0, 2, size=data_size)

# --- Categorized Humidity Level ---
# Scale 1: 80-100% (Very High)
# Scale 2: 60-79% (High)
# Scale 3: 40-59% (Moderate/Comfortable)
# Scale 4: 20-39% (Low)
# Scale 5: 0-19%  (Very Low)
humidity_scale = []
for h in raw_humidity_for_calc: # Use the temporary raw value for calculation
    if h >= 80:
        humidity_scale.append(1)
    elif h >= 60:
        humidity_scale.append(2)
    elif h >= 40:
        humidity_scale.append(3)
    elif h >= 20:
        humidity_scale.append(4)
    else:
        humidity_scale.append(5)

humidity_scale = np.array(humidity_scale)

# ---------------------------
# 3. ILLNESS / COMPLICATION
# ---------------------------
complication = np.random.randint(0, 3, size=data_size)
complication_adjustments = np.array([0, 300, 600])

# ---------------------------
# 4. NEW ACTIVITY SYSTEM
# ---------------------------
activity_type = np.random.randint(0, 6, size=data_size)

duration_minutes = []
pace = [] 
terrain_type = []
sweat_level = []

for a in activity_type:
    if a == 0: # walking
        duration_minutes.append(np.random.randint(10, 90))
        pace.append(np.random.uniform(3, 6))
    elif a == 1: # running
        duration_minutes.append(np.random.randint(10, 60))
        pace.append(np.random.uniform(6, 12))
    elif a == 2: # cycling
        duration_minutes.append(np.random.randint(20, 120))
        pace.append(np.random.uniform(10, 28))
    elif a == 3: # gym workout
        duration_minutes.append(np.random.randint(20, 90))
        pace.append(np.random.uniform(2, 5)) 
    elif a == 4: # yoga
        duration_minutes.append(np.random.randint(15, 60))
        pace.append(np.random.uniform(2, 4))
    elif a == 5: # sports
        duration_minutes.append(np.random.randint(20, 120))
        pace.append(np.random.uniform(4, 10))

    terrain_type.append(np.random.randint(0, 3))
    sweat_level.append(np.random.randint(0, 4))

duration_minutes = np.array(duration_minutes)
pace = np.array(pace, dtype=float)
terrain_type = np.array(terrain_type)
sweat_level = np.array(sweat_level)

# ---------------------------
# 5. INTENSITY SCORE
# ---------------------------
activity_base_map = {0: 0.10, 1: 0.30, 2: 0.25, 3: 0.20, 4: 0.05, 5: 0.30}
activity_base = np.array([activity_base_map[a] for a in activity_type])

pace_multiplier = np.zeros(data_size)
for i in range(data_size):
    a = activity_type[i]
    p = pace[i]
    if a == 0: pace_multiplier[i] = 0.05 if p < 4 else 0.10 if p < 5 else 0.20
    elif a == 1: pace_multiplier[i] = 0.20 if p < 8 else 0.30 if p < 10 else 0.40
    elif a == 2: pace_multiplier[i] = 0.10 if p < 16 else 0.20 if p < 22 else 0.30
    elif a == 3: pace_multiplier[i] = 0.05 
    elif a == 4: pace_multiplier[i] = 0.02
    elif a == 5: pace_multiplier[i] = 0.10 if p < 6 else 0.20 if p < 8 else 0.30

terrain_multiplier = np.where(terrain_type == 1, 0.10, np.where(terrain_type == 2, 0.05, 0.00))
sweat_multiplier = sweat_level * 0.05
duration_component = duration_minutes * 0.003

intensity_score = activity_base + pace_multiplier + terrain_multiplier + sweat_multiplier + duration_component
intensity_score = np.clip(intensity_score, 0, 1)

# ---------------------------
# 6. WATER INTAKE (UPDATED WITH IMAGE LOGIC)
# ---------------------------
BASE_ML_PER_KG = 30.0
base_intake = weight * BASE_ML_PER_KG
activity_adjustment = intensity_score * 1500
base_intake += complication_adjustments[complication] + activity_adjustment

stress_factor = np.zeros(data_size)
stress_factor += np.maximum(0, temperature - 25) * 0.01
stress_factor += is_direct_sun * 0.20
stress_factor += is_windy_or_fanned * 0.10
stress_factor -= is_indoors * 0.15

# --- APPLYING HUMIDITY SCALE RISKS ---
# This remains the same as it correctly calculates the stress factor based on the scale
humidity_stress = np.zeros(data_size)
for i in range(data_size):
    h_scale = humidity_scale[i]
    if h_scale == 1:
        humidity_stress[i] = 0.15 
    elif h_scale == 2:
        humidity_stress[i] = 0.05
    elif h_scale == 3:
        humidity_stress[i] = 0.00
    elif h_scale == 4:
        humidity_stress[i] = 0.10
    elif h_scale == 5:
        humidity_stress[i] = 0.20

stress_factor += humidity_stress

water_intake = np.maximum(base_intake * (1 + stress_factor), 2000)
water_intake += np.random.normal(0, 100, size=data_size)

# ---------------------------
# 7. CREATE DATAFRAME (CRITICAL CHANGE HERE)
# ---------------------------
df = pd.DataFrame({
    "age": age,
    "gender": gender,
    "weight": weight,
    "humidity_scale": humidity_scale, 
    "temperature": temperature,
    "complication": complication,
    "is_indoors": is_indoors,
    "is_ground_wet": is_ground_wet,
    "is_windy_or_fanned": is_windy_or_fanned,
    "is_direct_sun": is_direct_sun,
    "activity_type": activity_type,
    "duration_minutes": duration_minutes,
    "pace": pace,
    "terrain_type": terrain_type,
    "sweat_level": sweat_level,
    "intensity_score": intensity_score,
    "water_intake": water_intake.round(0)
})

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
random_filename = f"maruf_{timestamp}_{random_suffix}.csv"

df.to_csv(random_filename, index=False)
print(f"✅ Generated '{random_filename}' with **16 features (scale only)**!")