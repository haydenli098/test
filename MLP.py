import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pickle

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_FILE = r'c:\Users\flami\Documents\VSCode\ENGG2112\apsim_training_data2.parquet'
AVAILABLE_CROPS = ["Wheat", "Barley", "Canola", "Chickpea", "Oats", "Soybean"]
ROTATION_LENGTH = 5
MAP = r'c:\Users\flami\Documents\VSCode\ENGG2112\final_dataset.csv'

PRICES = {
    "Oats": 0.33,
    "Barley": 0.36,
    "Wheat": 0.39,
    "Chickpea": 0.80,
    "Canola": 0.78,
    "Soybean": 1.05
}

def one_hot_encode_sequence(seq_string):
    """Converts a sequence like 'Wheat, Canola...' into a binary neural network tensor."""
    crops = [c.strip() for c in seq_string.split(",")]
    encoded = []
    for crop in crops:
        vec = [1 if crop == c else 0 for c in AVAILABLE_CROPS]
        encoded.extend(vec)
    return encoded

def fix_map(map):
    """Cleans the map CSV to ensure it has the correct columns and formats."""
    df = pd.read_csv(map)
    
    # Check for required columns
    required_cols = ['latitude', 'longitude', 'rainfall_interpolated', 'elevation_m', 'soildepth_m', 'drainage', 'permeability', 'ph', 'nitrogen_interpolated']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Map CSV must contain the following columns: {required_cols}")
    
    # Handle missing values by filling with column means (or a default value)
    for col in required_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)
    
    return df

# ==========================================
# 2. PYTORCH MODEL (THE DIGITAL TWIN)
# ==========================================
class CropRotationPredictor(nn.Module):
    def __init__(self, num_env_features, num_crops, rotation_length, num_targets=1):
        super(CropRotationPredictor, self).__init__()
        input_size = num_env_features + (num_crops * rotation_length)
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_targets) # Output: Yield prediction(s)
        )
        
    def forward(self, x):
        return self.network(x)

# ==========================================
# 3. TRAINING ROUTINE
# ==========================================
def train_model():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Cannot find data file: {DATA_FILE}")
        
    df = pd.read_parquet(DATA_FILE)
    print(f"Loaded {len(df)} total samples from APSIM Parquet dataset.")
    
    # --- FIX: Filter out zero-yield scenarios ---
    # If the dataset has many crop failures (yield=0), it can skew the model's
    # predictions towards zero. For an optimization task, we are more interested
    # in what makes a GOOD yield, so we can train only on successful runs.
    initial_count = len(df)
    df = df[df['TotalYield'] > 0.1] # Use a small threshold to avoid floating point issues
    print(f"Filtered out zero-yield runs. Retaining {len(df)} of {initial_count} successful samples for training.")

    # Extract Environmental Features
    env_cols = ['rainfall_interpolated', 'elevation_m', 'soildepth_m', 'drainage', 'permeability', 'ph', 'nitrogen']
    env_features = df[env_cols].values
    
    # Extract Sequence Features
    seq_features = np.array([one_hot_encode_sequence(seq) for seq in df['Rotation']])
    
    # Combine inputs and format Target (TotalYield)
    X = np.hstack((env_features, seq_features))
    
    # Target variables (sequence of yearly yields instead of just cumulative)
    target_cols = [f'Yield_Yr{i+1}' for i in range(ROTATION_LENGTH)]
    if all(col in df.columns for col in target_cols):
        y = df[target_cols].values
        num_targets = ROTATION_LENGTH
    else:
        print("⚠️ Yearly yield columns not found! Regenerate the dataset. Training on TotalYield instead.")
        y = df['TotalYield'].values.reshape(-1, 1)
        num_targets = 1
    
    # Scale Data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    # Use MinMaxScaler for Yield. Yield is strictly non-negative, 
    # and StandardScaler heavily encourages negative predictions for poor environments.
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize Model
    model = CropRotationPredictor(num_env_features=len(env_cols), num_crops=len(AVAILABLE_CROPS), rotation_length=ROTATION_LENGTH, num_targets=num_targets)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 100
    print(f"Training MLP Surrogate Model for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 25 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}")
            
    print("Training Complete!")
    
    # --- Evaluate Accuracy on Test Set ---
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        scaled_predictions = model(X_test_tensor).numpy()
        
        # Inverse transform to get real-world units (kg/ha) for a fair comparison
        actual_yields = scaler_y.inverse_transform(y_test)
        predicted_yields = scaler_y.inverse_transform(scaled_predictions)
        
        # Calculate R-squared score on the real-world values (1.0 is perfect)
        r2 = r2_score(actual_yields, predicted_yields)
        # Calculate MAE in real-world units (kg/ha)
        mae = mean_absolute_error(actual_yields, predicted_yields)
        # Calculate RMSE in real-world units (kg/ha)
        rmse = np.sqrt(mean_squared_error(actual_yields, predicted_yields))

        print(f"\n--- MODEL ACCURACY ---")
        print(f"Test R-squared (R²): {r2:.4f}")
        print(f"Mean Absolute Error: {mae:.2f} kg/ha off on average")
        print(f"Root Mean Squared Error: {rmse:.2f} kg/ha off on average")
        print(f"----------------------\n")
        
        # Plot Actual vs Predicted
        if num_targets > 1:
            actual_plot = actual_yields.sum(axis=1)
            predicted_plot = predicted_yields.sum(axis=1)
        else:
            actual_plot = actual_yields
            predicted_plot = predicted_yields
            
        plt.figure(figsize=(8, 6))
        plt.scatter(actual_plot, predicted_plot, alpha=0.5, color='blue')
        
        # Plot perfect prediction line
        def plot_perfect_line():
            min_val = min(actual_plot.min(), predicted_plot.min())
            max_val = max(actual_plot.max(), predicted_plot.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            plt.xlabel('Actual Total Yield (kg/ha)')
            plt.ylabel('Predicted Total Yield (kg/ha)')
            plt.title('Actual vs Predicted Total Yield')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        # plot_perfect_line()
    # Save model & scalers
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_env_features': len(env_cols),
        'num_crops': len(AVAILABLE_CROPS),
        'rotation_length': ROTATION_LENGTH,
        'num_targets': num_targets
    }, 'mlp_model_full.pth')
    with open('mlp_scaler_x.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    with open('mlp_scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)

    return model, scaler_X, scaler_y

# ==========================================
# 3.5 LOAD OR TRAIN
# ==========================================
def load_or_train_model():
    if os.path.exists('mlp_model_full.pth') and os.path.exists('mlp_scaler_x.pkl') and os.path.exists('mlp_scaler_y.pkl'):
        print("Loading existing MLP model and scalers...")
        checkpoint = torch.load('mlp_model_full.pth')
        model = CropRotationPredictor(
            num_env_features=checkpoint['num_env_features'],
            num_crops=checkpoint['num_crops'],
            rotation_length=checkpoint['rotation_length'],
            num_targets=checkpoint['num_targets']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        with open('mlp_scaler_x.pkl', 'rb') as f:
            scaler_X = pickle.load(f)
        with open('mlp_scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
        return model, scaler_X, scaler_y
    else:
        print("Saved model not found. Training from scratch...")
        return train_model()
    
    
# ==========================================
# 4. OPTIMIZATION LOOP
# ==========================================
def optimize_rotation(model, scaler_X, scaler_y, target_env, iterations=5000):
    model.eval()
    
    best_rotation = None
    best_revenue = -float('inf')
    best_yields = None
    
    print(f"\n--- Optimizing for Paddock Location [Rain={target_env[0]}mm, Elev={target_env[1]}m, Depth={target_env[2]}m, Drain={target_env[3]}, Perm={target_env[4]}, pH={target_env[5]}, N={target_env[6]}] ---")
    
    with torch.no_grad():
        for _ in range(iterations):
            proposed_rotation = [random.choice(AVAILABLE_CROPS) for _ in range(ROTATION_LENGTH)]
            rot_str = ", ".join(proposed_rotation)
            
            encoded_rot = one_hot_encode_sequence(rot_str)
            X_input = np.hstack((target_env, encoded_rot)).reshape(1, -1)
            X_input_scaled = scaler_X.transform(X_input)
            
            # Predict
            pred_scaled = model(torch.FloatTensor(X_input_scaled))
            pred_yields = scaler_y.inverse_transform(pred_scaled.numpy())[0]
            
            pred_revenue = sum(pred_yields[i] * PRICES.get(proposed_rotation[i], 0.0) for i in range(ROTATION_LENGTH))
            
            if pred_revenue > best_revenue:
                best_revenue = pred_revenue
                best_rotation = rot_str
                best_yields = pred_yields
                
    return best_rotation, max(0.0, best_revenue), best_yields

def get_enviro_data(lat, long):
    lat = float(lat)
    long = float(long)
    
    # Load and clean the dataset
    df = fix_map(MAP)
    
    # Find the nearest location using Euclidean distance
    distances = (df['latitude'] - lat)**2 + (df['longitude'] - long)**2
    closest_idx = distances.idxmin()
    closest_site = df.loc[closest_idx]
    
    print(f"Matched input to closest known site: Latitude {closest_site['latitude']:.4f}, Longitude {closest_site['longitude']:.4f}")
    
    # --- FIX: Map features to APSIM training domains ---
    # The neural network was trained on synthetic APSIM bounds, but the map uses different units/classes.
    # We must translate the map values into the ranges the model understands.
    
    mapped_rain = closest_site['rainfall_interpolated'] * ROTATION_LENGTH
    
    # Drainage: Map class (1-6) -> APSIM SWCON (0.3-0.7)
    mapped_drainage = max(0.3, min(0.7, 0.3 + ((closest_site['drainage'] - 1) / 5) * 0.4))
    
    # Permeability: Map class (1-4) -> APSIM KS (20.0-80.0)
    mapped_perm = max(20.0, min(80.0, 20.0 + ((closest_site['permeability'] - 1) / 3) * 60.0))
    
    # Nitrogen: Map Total N (%) -> APSIM Initial N kg/ha (approximate mapping to 30-150 range)
    mapped_n = max(30.0, min(150.0, closest_site['nitrogen_interpolated'] * 150.0))
    
    # FIX: SoilDepth was a constant in the training data, but Elevation was randomized.
    # Use the actual elevation from the map data instead of a hardcoded value.
    mapped_elevation = closest_site['elevation_m']
    mapped_soildepth = 1.2
    
    # Extract features matching the model's exact expected order:
    # ['rainfall_interpolated', 'elevation_m', 'soildepth_m', 'drainage', 'permeability', 'ph', 'nitrogen']
    return [
        mapped_rain, mapped_elevation, mapped_soildepth, mapped_drainage, mapped_perm, closest_site['ph'], mapped_n
    ]


if __name__ == "__main__":
    trained_model, scaler_x, scaler_y = load_or_train_model()
    location = input("Enter location (latitude, longitude): ")
    enviro_data = get_enviro_data(location.split(',')[0], location.split(',')[1])

    best_seq, expected_revenue, best_yields = optimize_rotation(trained_model, scaler_x, scaler_y, enviro_data, iterations=5000)

    total_yield = sum(best_yields)
    per_annum_yield = total_yield / ROTATION_LENGTH
    
    print(f"\n✅ OPTIMAL 5-YEAR ROTATION: {best_seq}")
    print(f"📊 Expected Total Yield: {total_yield:.2f} kg/ha")
    print(f"📈 Expected Per Annum Yield: {per_annum_yield:.2f} kg/ha")
    print(f"💰 PREDICTED REVENUE: ${expected_revenue:.2f} / ha")
    print(f"\nYearly Breakdown:")
    crops = [c.strip() for c in best_seq.split(",")]
    for year, (crop, yield_val) in enumerate(zip(crops, best_yields), 1):
        print(f"  Year {year} ({crop}): {yield_val:.2f} kg/ha")
