import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import uuid
import argparse

# -----------------------------
# Configuration
# -----------------------------
MODEL_FILENAME_PREFIX = "maruf_"
DEFAULT_DATASET_FILE = "maruf_20251128115103_0rzgk9.csv"

# Features must match CSV exactly
# CRITICAL: REMOVE "humidity" from the list (using humidity_scale)
FEATURE_COLS = [
    "age",
    "gender",
    "weight",
    "humidity_scale", 
    "temperature",
    "complication",
    "is_indoors",
    "is_ground_wet",
    "is_windy_or_fanned",
    "is_direct_sun",
    "activity_type",
    "duration_minutes",
    "pace",
    "terrain_type",
    "sweat_level",
    "intensity_score"
]

INPUT_DIMENSION = len(FEATURE_COLS)

def train_model(dataset_file=DEFAULT_DATASET_FILE):
    random_filename_model = MODEL_FILENAME_PREFIX + str(uuid.uuid4()) + ".h5"
    random_filename_scaler = MODEL_FILENAME_PREFIX + str(uuid.uuid4()) + ".pkl"
    
    # -----------------------------
    # Load dataset
    # -----------------------------
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"CSV file not found: {dataset_file}. Please check the filename.")

    df = pd.read_csv(dataset_file)
    print(f"✅ Loaded dataset: {dataset_file}")

    # Validate columns
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    X = df[FEATURE_COLS].values
    y = df["water_intake"].values

    # -----------------------------
    # Train/Test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # Feature scaling
    # -----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------
    # Model definition
    # -----------------------------
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu", input_shape=(INPUT_DIMENSION,)),
        tf.keras.layers.Dropout(0.2), # Added Dropout for regularization
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="relu") 
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()

    # -----------------------------
    # Training
    # -----------------------------
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        restore_best_weights=True,
        verbose=1
    )
    
    # We save the BEST model only
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        random_filename_model, 
        monitor='val_loss', 
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        X_train_scaled, y_train,
        epochs=300, # Increased epochs since we have early stopping
        batch_size=32,
        validation_data=(X_test_scaled, y_test),
        verbose=1,
        callbacks=[early_stopping, checkpoint]
    )

    # -----------------------------
    # Evaluation
    # -----------------------------
    # Load best model for evaluation (Checkpoint saves it)
    best_model = tf.keras.models.load_model(random_filename_model)
    loss, mae = best_model.evaluate(X_test_scaled, y_test, verbose=0)
    
    print(f"\n✅ Model Evaluation (Best Model):")
    print(f"  MSE: {loss:.2f}")
    print(f"  MAE: {mae:.2f} ml")

    # -----------------------------
    # Save Scaler
    # -----------------------------
    joblib.dump(scaler, random_filename_scaler)

    print(f"\n✅ Model saved to: {random_filename_model}")
    print(f"✅ Scaler saved to: {random_filename_scaler}")
    
    print("\n⚠️  IMPORTANT: Please update 'config.py' with these new filenames!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the hydration prediction model.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_FILE, help="Path to the CSV dataset file.")
    args = parser.parse_args()
    
    train_model(dataset_file=args.dataset)