
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress sklearn feature names warning
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# =========================
# GLOBAL STATE
# =========================
_model = None
_spatial_tree = None
_soil_data_np = None
_soil_features = None
_df = None

def _initialize_model():
    """Initialize the model lazily on first use"""
    global _model, _spatial_tree, _soil_data_np, _soil_features, _df
    
    if _model is not None:
        return  # Already initialized
    
    # =========================
    # 1. LOAD + SPLIT DATA
    # =========================
    _df = pd.read_csv('final_crop_rotation_plan.csv')

    _soil_features = [
        'Elevation_m', 'SoilDepth_m', 'Drainage',
        'Permeability', 'pH', 'OrganicCarbon_pct'
    ]

    # Split dataset
    df_train, df_temp = train_test_split(_df, test_size=0.30, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=42)

    # Feature / Target split
    X_train, y_train = df_train[_soil_features], df_train['Recommended_Rotation']
    X_val, y_val = df_val[_soil_features], df_val['Recommended_Rotation']
    X_test, y_test = df_test[_soil_features], df_test['Recommended_Rotation']

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    # =========================
    # 2. MODEL TRAINING + TUNING
    # =========================
    smoothing_options = [1e-9, 1e-5, 1e-1, 1.0]

    best_accuracy = 0
    best_model = None

    print("\n--- Validation Phase ---")
    for alpha in smoothing_options:
        model = GaussianNB(var_smoothing=alpha)
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)

        print(f"Alpha: {alpha} → Val Accuracy: {val_acc:.2%}")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model = model

    print("\n✅ Best model selected!")
    _model = best_model

    # =========================
    # 3. SPATIAL INDEX (FAST LOOKUP)
    # =========================
    coordinates = _df[['Latitude', 'Longitude']].values
    _spatial_tree = KDTree(coordinates)

    # Pre-extract soil features as NumPy (faster access)
    _soil_data_np = _df[_soil_features].values

def predict_crop_rotation_by_coordinates(latitude, longitude):
    """Predict crop rotation for given coordinates"""
    _initialize_model()
    
    _, idx = _spatial_tree.query([latitude, longitude])
    input_vector = _soil_data_np[idx].reshape(1, -1)
    return _model.predict(input_vector)[0]

# Only run this if NB.py is executed directly (not imported)
if __name__ == "__main__":
    _initialize_model()
    
    # =========================
    # 4. PIPELINE TESTING
    # =========================
    print("\n=== SYSTEM-WIDE PIPELINE TEST ===")

    df_train, df_temp = train_test_split(_df, test_size=0.30, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=42)
    X_test, y_test = df_test[_soil_features], df_test['Recommended_Rotation']
    
    test_coords = df_test[['Latitude', 'Longitude']].values
    pipeline_test_preds = [
        predict_crop_rotation_by_coordinates(lat, lon)
        for lat, lon in test_coords
    ]

    pipeline_acc = accuracy_score(y_test, pipeline_test_preds)
    print(f"Pipeline Accuracy: {pipeline_acc:.2%}")

    # =========================
    # 5. RANDOM SPOT CHECKS
    # =========================
    print("\n=== RANDOM SPOT CHECKS ===")

    sample_rows = df_test.sample(5, random_state=42)

    for i, row in enumerate(sample_rows.itertuples(index=False), 1):
        pred = predict_crop_rotation_by_coordinates(row.Latitude, row.Longitude)

        print(f"\nTest #{i}")
        print(f"Coordinates: ({row.Latitude:.4f}, {row.Longitude:.4f})")
        print(f"True: {row.Recommended_Rotation}")
        print(f"Pred: {pred}")
        print(f"Match: {'✅' if pred == row.Recommended_Rotation else '❌'}")

    # =========================
    # 6. USER INPUT TOOL
    # =========================
    print("\n--- 🌾 CROP ROTATION CALCULATOR 🌾 ---")

    try:
        user_input = input("Enter Latitude & Longitude (format: '-30, 150' or enter separately): ").strip()
        
        if ',' in user_input:
            parts = user_input.split(',')
            user_lat = float(parts[0].strip())
            user_lon = float(parts[1].strip())
        else:
            user_lat = float(user_input)
            user_lon = float(input("Enter Longitude: "))

        print("\nProcessing...")

        predicted_plan = predict_crop_rotation_by_coordinates(user_lat, user_lon)

        print("\n--------------------------------")
        print(f"Location: ({user_lat:.4f}, {user_lon:.4f})")
        print(f"Recommended Rotation: {predicted_plan}")
        print("--------------------------------")

    except ValueError as e:
        print(f"❌ Input Error: Please enter valid numbers. Example: -30, 150")
    except Exception as e:
        print(f"❌ Error: {e}")
