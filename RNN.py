import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
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

CROP_N_IMPACT = {
    "Wheat": -25.0,
    "Barley": -20.0,
    "Canola": -30.0,
    "Oats": -15.0,
    "Chickpea": 10.0,
    "Soybean": 15.0
}

def one_hot_encode_sequence(seq_string):
    """Converts a sequence like 'Wheat, Canola...' into a binary neural network tensor."""
    crops = [c.strip() for c in seq_string.split(",")]
    encoded = []
    for crop in crops:
        vec = [1 if crop == c else 0 for c in AVAILABLE_CROPS]
        encoded.extend(vec)
    return encoded

def fix_map(map_path):
    """Cleans the map CSV to ensure it has the correct columns and formats."""
    df = pd.read_csv(map_path)
    
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
# 2. PYTORCH MODEL (THE DIGITAL TWIN - RNN)
# ==========================================
class CropRotationRNN(nn.Module):
    def __init__(self, num_env_features, num_crops, rotation_length, num_targets=1, hidden_size=128, num_layers=3):
        super(CropRotationRNN, self).__init__()
        self.num_env_features = num_env_features
        self.num_crops = num_crops
        self.rotation_length = rotation_length
        self.num_targets = num_targets
        
        # Input per timestep: env features + single crop one-hot vector
        input_size = num_env_features + num_crops
        
        # Recurrent Layer (LSTM is generally better at capturing long-term dependencies than a plain RNN)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2, bidirectional=True)
        
        # Output Layer maps the hidden state at each timestep to a yield prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x is now expected to be shape (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        
        # Pass each timestep's hidden state through FC to get yearly yield predictions
        predictions = self.fc(out).squeeze(-1) # shape: (batch, seq_len)
        
        # If the model expects a single TotalYield output instead of yearly yields, sum across the timesteps
        if self.num_targets == 1:
            return predictions.sum(dim=1, keepdim=True)
            
        return predictions

# ==========================================
# 3. TRAINING ROUTINE
# ==========================================
def train_model():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Cannot find data file: {DATA_FILE}")
        
    df = pd.read_parquet(DATA_FILE)
    print(f"Loaded {len(df)} total samples from APSIM Parquet dataset.")

    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42).reset_index(drop=True)
        print("Randomly selected 10,000 samples for training.")

    initial_count = len(df)
    df = df[df['TotalYield'] > 0.1]
    print(f"Filtered out zero-yield runs. Retaining {len(df)} of {initial_count} successful samples for training.")

    # Extract Static Environmental Features
    static_env_cols = ['elevation_m', 'soildepth_m', 'drainage', 'permeability', 'ph']
    static_env_features = df[static_env_cols].values

    seq_features = np.array([one_hot_encode_sequence(seq) for seq in df['Rotation']]).reshape(-1, ROTATION_LENGTH, len(AVAILABLE_CROPS))
    
    # Create Stepwise Environmental Data dynamically to incorporate yearly rainfall
    env_features_stepwise = np.zeros((len(df), ROTATION_LENGTH, len(static_env_cols) + 2))
    
    rain_cols = [f'Rain_Yr{i+1}' for i in range(ROTATION_LENGTH)]
    if all(col in df.columns for col in rain_cols):
        yearly_rain = df[rain_cols].values
    else:
        print("⚠️ Yearly rain columns not found! Falling back to average annual rainfall.")
        yearly_rain = np.repeat((df['rainfall_interpolated'] / ROTATION_LENGTH).values[:, np.newaxis], ROTATION_LENGTH, axis=1)

    n_cols = [f'N_Yr{i+1}' for i in range(ROTATION_LENGTH)]
    if all(col in df.columns for col in n_cols):
        yearly_n = df[n_cols].values
    else:
        print("⚠️ Yearly N columns not found! Falling back to static nitrogen.")
        yearly_n = np.repeat(df['nitrogen'].values[:, np.newaxis], ROTATION_LENGTH, axis=1)

    for i in range(ROTATION_LENGTH):
        env_features_stepwise[:, i, 0] = yearly_rain[:, i]      # Feature 0: Rain for this specific year
        env_features_stepwise[:, i, 1:6] = static_env_features  # Features 1-5: Static environment
        env_features_stepwise[:, i, 6] = yearly_n[:, i]         # Feature 6: Dynamic Nitrogen
    
    # Combine inputs: shape (N, 5, 13)
    X = np.concatenate((env_features_stepwise, seq_features), axis=2)
    target_cols = [f'Yield_Yr{i+1}' for i in range(ROTATION_LENGTH)]
    if all(col in df.columns for col in target_cols):
        y = df[target_cols].values
        num_targets = ROTATION_LENGTH
    else:
        print("⚠️ Yearly yield columns not found! Regenerate the dataset. Training on TotalYield instead.")
        y = df['TotalYield'].values.reshape(-1, 1)
        num_targets = 1
    
    scaler_X = StandardScaler()
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    # Scale 3D data by flattening it to 2D first
    X_flat = X.reshape(-1, X.shape[2])
    X_scaled_flat = scaler_X.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape(X.shape)
    
    y_scaled = scaler_y.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = CropRotationRNN(
        num_env_features=7, 
        num_crops=len(AVAILABLE_CROPS), 
        rotation_length=ROTATION_LENGTH, 
        num_targets=num_targets
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    epochs = 200
    print(f"Training RNN Surrogate Model for {epochs} epochs (with early stopping/scheduler)...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step(epoch_loss)
            
        if (epoch+1) % 25 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}")
            
    print("Training Complete!")
    
    # --- Evaluate Accuracy on Test Set ---
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        scaled_predictions = model(X_test_tensor).numpy()
        
        actual_yields = scaler_y.inverse_transform(y_test)
        predicted_yields = scaler_y.inverse_transform(scaled_predictions)
        
        r2 = r2_score(actual_yields, predicted_yields)
        mae = mean_absolute_error(actual_yields, predicted_yields)
        rmse = np.sqrt(mean_squared_error(actual_yields, predicted_yields))

        print(f"\n--- MODEL ACCURACY ---")
        print(f"Test R-squared (R²): {r2:.4f}")
        print(f"Mean Absolute Error: {mae:.2f} kg/ha off on average")
        print(f"Root Mean Squared Error: {rmse:.2f} kg/ha off on average")
        print(f"----------------------\n")
        
    # Save model and scalers
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_env_features': 7,
        'num_crops': len(AVAILABLE_CROPS),
        'rotation_length': ROTATION_LENGTH,
        'num_targets': num_targets,
        'hidden_size': 128,
        'num_layers': 3
    }, 'rnn_model_full.pth')
    with open('rnn_scaler_x.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    with open('rnn_scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)

    return model, scaler_X, scaler_y

# ==========================================
# 3.5 LOAD OR TRAIN
# ==========================================
def load_or_train_model():
    if os.path.exists('rnn_model_full.pth') and os.path.exists('rnn_scaler_x.pkl') and os.path.exists('rnn_scaler_y.pkl'):
        print("Loading existing RNN model and scalers...")
        checkpoint = torch.load('rnn_model_full.pth')
        model = CropRotationRNN(
            num_env_features=checkpoint['num_env_features'],
            num_crops=checkpoint['num_crops'],
            rotation_length=checkpoint['rotation_length'],
            num_targets=checkpoint['num_targets'],
            hidden_size=checkpoint.get('hidden_size', 128),
            num_layers=checkpoint.get('num_layers', 3)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        with open('rnn_scaler_x.pkl', 'rb') as f:
            scaler_X = pickle.load(f)
        with open('rnn_scaler_y.pkl', 'rb') as f:
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
            
            # Prepare stepwise environmental variables dynamically
            env_stepwise = np.zeros((ROTATION_LENGTH, 7))
            curr_n = target_env[6]
            for i in range(ROTATION_LENGTH):
                crop = proposed_rotation[i]
                env_stepwise[i] = [
                    target_env[0], target_env[1], target_env[2], 
                    target_env[3], target_env[4], target_env[5], 
                    curr_n
                ]
                # Simulate Nitrogen draw-down
                curr_n = max(5.0, curr_n + CROP_N_IMPACT.get(crop, -20.0))
                
            encoded_rot = np.array(one_hot_encode_sequence(rot_str)).reshape(ROTATION_LENGTH, -1)
            
            X_input = np.hstack((env_stepwise, encoded_rot))
            X_input_scaled = scaler_X.transform(X_input)
            
            X_tensor = torch.FloatTensor(X_input_scaled).unsqueeze(0)
            pred_scaled = model(X_tensor)
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
    
    # Use the annual rainfall data directly for our step-wise predictions
    mapped_rain = closest_site['rainfall_interpolated']
    
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
    
    # Try to read a random location from the dataset
    try:
        dataset = pd.read_csv(MAP)
        if 'latitude' not in dataset.columns or 'longitude' not in dataset.columns:
            raise ValueError("Dataset must contain 'latitude' and 'longitude' columns")
        
        random_row = dataset.sample(n=1).iloc[0]
        lat = random_row['latitude']
        lon = random_row['longitude']
        print(f"Using random location from dataset: Latitude {lat}, Longitude {lon}")
    except Exception as e:
        print(f"Warning: Could not load random location from dataset: {e}")
        print("Please enter location manually.")
        location_input = input("Enter location (latitude, longitude): ").strip()
        parts = location_input.split(',')
        if len(parts) != 2:
            raise ValueError(f"Invalid location format. Expected 'lat,lon' but got: {location_input}")
        lat, lon = parts[0].strip(), parts[1].strip()
    
    enviro_data = get_enviro_data(lat, lon)

    best_seq, expected_revenue, best_yields = optimize_rotation(trained_model, scaler_x, scaler_y, enviro_data, iterations=5000)

    print(f"\n✅ OPTIMAL 5-YEAR ROTATION: {best_seq}")
    print(f"📈 PREDICTED REVENUE: ${expected_revenue:.2f} / ha")
