import os
import random
import json
import tempfile
import shutil
import numpy as np
import pandas as pd
import traceback
from apsimNGpy.core.apsim import ApsimModel

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_APSIMX = "base_simulation.apsimx"
MAP_CSV = "final_dataset.csv"
ROTATION_LENGTH = 5

def get_enviro_data(lat, lon):
    """Fetches mapped environmental data using latitude and longitude."""
    df = pd.read_csv(MAP_CSV)
    required_cols = ['latitude', 'longitude', 'rainfall_interpolated', 'elevation_m', 'soildepth_m', 'drainage', 'permeability', 'ph', 'nitrogen_interpolated']
    
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Map CSV must contain the following columns: {required_cols}")
        
    for col in required_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)

    # Find closest coordinate via Euclidean distance
    distances = (df['latitude'] - lat)**2 + (df['longitude'] - lon)**2
    closest_idx = distances.idxmin()
    closest_site = df.loc[closest_idx]
    
    print(f"Matched input to closest known site: Latitude {closest_site['latitude']:.4f}, Longitude {closest_site['longitude']:.4f}")
    
    mapped_rain = closest_site['rainfall_interpolated']
    mapped_drainage = max(0.3, min(0.7, 0.3 + ((closest_site['drainage'] - 1) / 5) * 0.4))
    mapped_perm = max(20.0, min(80.0, 20.0 + ((closest_site['permeability'] - 1) / 3) * 60.0))
    mapped_n = max(30.0, min(150.0, closest_site['nitrogen_interpolated'] * 150.0))
    
    return {
        'rainfall': mapped_rain,
        'elevation': closest_site['elevation_m'],
        'soildepth': 1.2,
        'drainage': mapped_drainage,
        'permeability': mapped_perm,
        'ph': closest_site['ph'],
        'nitrogen': mapped_n,
        'carbon': 1.5,
        'tmin': closest_site['tmin_interpolated'],
        'tmax': closest_site['tmax_interpolated']
    }

def modify_met_with_random_temps(base_met_path, tmin, tmax, rotation_length, sim_id):
    """
    Modifies a .met file to use random temperatures between tmin and tmax for all days.
    Returns the path to the modified temporary .met file.
    """
    try:
        with open(base_met_path, 'r') as f:
            lines = f.readlines()
        
        # Find where data starts
        data_start = next((i for i, line in enumerate(lines) if line.strip().lower().startswith('year')), 0)
        
        # Build header with adjusted tav (average temperature)
        tavg = (tmin + tmax) / 2.0
        header_lines = []
        for line in lines[:data_start]:
            if line.strip().lower().startswith('tav'):
                try:
                    parts = line.split('=')
                    header_lines.append(f"tav = {tavg:.2f} (oC)\n")
                except:
                    header_lines.append(line)
            else:
                header_lines.append(line)
        
        # Add column headers
        header_lines.append(lines[data_start])
        if data_start + 1 < len(lines):
            header_lines.append(lines[data_start + 1])
        
        # Read data
        df_met = pd.read_csv(base_met_path, skiprows=data_start, sep=r'\s+', engine='python', on_bad_lines='skip')
        df_met.columns = df_met.columns.str.lower()
        df_met = df_met.iloc[1:].copy()
        df_met = df_met.dropna(subset=['year', 'day'])
        df_met['year'] = pd.to_numeric(df_met['year'], errors='coerce').astype(int)
        df_met['day'] = pd.to_numeric(df_met['day'], errors='coerce').astype(int)
        
        # Generate random temperatures for each day
        if 'maxt' in df_met.columns:
            df_met['maxt'] = [round(random.uniform(tmin, tmax), 2) for _ in range(len(df_met))]
        if 'mint' in df_met.columns:
            df_met['mint'] = [round(random.uniform(tmin, tmax * 0.8), 2) for _ in range(len(df_met))]  # Minimum is slightly cooler
        
        # Create temporary met file
        temp_met_path = os.path.join(tempfile.gettempdir(), f"temp_weather_{sim_id}.met")
        with open(temp_met_path, 'w') as f:
            f.writelines(header_lines)
            df_met.to_csv(f, sep=' ', index=False, header=False, na_rep='?')
        
        print(f"Created modified met file with temperatures between {tmin:.1f}°C and {tmax:.1f}°C")
        return temp_met_path
    except Exception as e:
        print(f"Failed to modify met file: {e}")
        return None

def simulate_rotation_yield(rotation, enviro_data=None, lat=None, lon=None, sim_id="user_run"):
    """
    Runs an APSIM simulation for a given crop rotation sequence and environment.
    Environment can be passed as a dictionary, or resolved via lat/lon coordinates.
    """
    if enviro_data is None:
        if lat is not None and lon is not None:
            enviro_data = get_enviro_data(float(lat), float(lon))
        else:
            raise ValueError("Must provide either an enviro_data dictionary or lat/lon coordinates.")
            
    rainfall = enviro_data.get('rainfall', 500.0)
    drainage = enviro_data.get('drainage', 0.5)
    permeability = enviro_data.get('permeability', 50.0)
    ph = max(3.6, enviro_data.get('ph', 6.5)) # APSIM requires pH >= 3.5
    nitrogen = enviro_data.get('nitrogen', 50.0)
    carbon = enviro_data.get('carbon', 1.5)
    tmin = enviro_data.get('tmin', 5.0)
    tmax = enviro_data.get('tmax', 25.0)
    
    total_yield = 0.0
    yearly_yields = [0.0] * ROTATION_LENGTH
    temp_met_path = None
    
    try:
        model = ApsimModel(BASE_APSIMX, copy=True)
        
        # Find and modify the weather file with random temperatures
        # Look for a met file in the WeatherFiles directory
        weather_dir = r"c:\Users\flami\Documents\VSCode\ENGG2112\WeatherFiles"
        base_met = None
        if os.path.exists(weather_dir):
            met_files = [f for f in os.listdir(weather_dir) if f.endswith('.met')]
            if met_files:
                base_met = os.path.join(weather_dir, met_files[0])
        
        if base_met and os.path.exists(base_met):
            temp_met_path = modify_met_with_random_temps(base_met, tmin, tmax, ROTATION_LENGTH, sim_id)
            if temp_met_path:
                try:
                    model.replace_met_file(temp_met_path)
                except TypeError:
                    # Fallback: if replace_met_file doesn't accept arguments, 
                    # copy temp file to default location used by model
                    try:
                        import shutil
                        model_weather_dir = os.path.join(os.path.dirname(BASE_APSIMX), 'WeatherFiles')
                        if not os.path.exists(model_weather_dir):
                            os.makedirs(model_weather_dir)
                        fallback_met = os.path.join(model_weather_dir, f"modified_{sim_id}.met")
                        shutil.copy(temp_met_path, fallback_met)
                        temp_met_path = fallback_met  # Update path for cleanup
                        print(f"Copied modified met file to {fallback_met}")
                    except Exception as fallback_err:
                        print(f"Warning: Could not apply modified met file: {fallback_err}")

        # Modify Soil
        model.edit_model(model_type="Models.WaterModel.WaterBalance", model_name="SoilWater", SWCON=[drainage] * 7)
        model.replace_soil_property_values(parameter="KS", param_values=[permeability] * 7, soil_child="physical")
        model.replace_soil_property_values(parameter="PH", param_values=[ph] * 7, soil_child="chemical")
        model.replace_soil_property_values(parameter="Carbon", param_values=[carbon, carbon*0.8, carbon*0.5, 0.3, 0.2, 0.1, 0.1], soil_child="organic")

        # Modify Nitrogen
        no3_values = [nitrogen * 0.6, nitrogen * 0.2, nitrogen * 0.1, 5, 2, 1, 1]
        nh4_values = [5, 2, 1, 1, 1, 1, 1] 
        model.edit_model(model_type="Models.Soils.Solute", model_name="NO3", InitialValues=no3_values)
        model.edit_model(model_type="Models.Soils.Solute", model_name="NH4", InitialValues=nh4_values)
        
        # Set Crop Sequence
        model.edit_model(model_type="Models.Manager", model_name="RotationManager", CropSequence=rotation)

        print(f"Running APSIM Simulation for rotation: '{rotation}'...")
        model.run()

        # Extract Yield Results
        try:
            report_df = model.results
            if isinstance(report_df, dict) and report_df:
                # Find the dataframe containing yield data
                selected_df = None
                for key, df_val in report_df.items():
                    if isinstance(df_val, pd.DataFrame) and any('yield' in str(c).lower() for c in df_val.columns):
                        selected_df = df_val
                        break
                report_df = selected_df if selected_df is not None else list(report_df.values())[0]

            if report_df is not None and not isinstance(report_df, dict) and not report_df.empty:
                yield_cols = [c for c in report_df.columns if 'yield' in str(c).lower()]
                if yield_cols:
                    # Just grab raw totals across the sequence for simplicity
                    year_totals = report_df[yield_cols].sum(axis=1).tolist()
                    for i in range(min(len(year_totals), ROTATION_LENGTH)):
                        yearly_yields[i] = round(year_totals[i], 2)
                    total_yield = round(sum(yearly_yields), 2)
                else:
                    print(f"No yield columns found in results. Columns available: {list(report_df.columns)}")
        except Exception as db_err:
            print(f"Failed to fetch DB results: {db_err}")

    except Exception as e:
        print(f"Error during APSIM simulation execution: {e}")
        traceback.print_exc()
    finally:
        # Clean up temporary met file if created
        if temp_met_path and os.path.exists(temp_met_path):
            try:
                os.remove(temp_met_path)
            except:
                pass

    return {
        'Rotation': rotation,
        'Environment': enviro_data,
        'TotalYield': total_yield,
        'YearlyYields': yearly_yields
    }

def parse_enviro_input(user_input):
    """
    Automatically determines if the input string is Lat/Lon or raw environmental data (JSON).
    Returns a tuple: (enviro_data_dict, lat, lon).
    """
    user_input = user_input.strip()

    # 1. Check if it's raw environmental data (JSON dictionary)
    if user_input.startswith("{") and user_input.endswith("}"):
        try:
            enviro_data = json.loads(user_input)
            return enviro_data, None, None
        except json.JSONDecodeError:
            print("Warning: Input looked like JSON but failed to parse.")

    # 2. Check if it's a Lat/Lon pair
    try:
        parts = [p.strip() for p in user_input.split(',')]
        if len(parts) == 2:
            lat, lon = float(parts[0]), float(parts[1])
            return None, lat, lon
    except ValueError:
        pass

    raise ValueError("Invalid input format. Expected Lat/Lon (e.g. '-33.8, 151.2') or JSON string.")

if __name__ == "__main__":
    
    rotation_input = input("Enter crop rotation sequence (comma-separated, e.g., 'Wheat, Canola, Wheat, Chickpea, Barley') or press Enter to use default: ")
    user_data = input("Input Enviro data (Lat/Lon e.g., '-33.8, 151.2' OR raw JSON dict [rainfall, drainage, permeability, ph, nitrogen, carbon]) : ")
    if rotation_input.strip():
        example_rotation = rotation_input.strip()
    try:
        enviro_dict, parsed_lat, parsed_lon = parse_enviro_input(user_data)
        
        print(f"\n--- Simulating ---")
        result = simulate_rotation_yield(example_rotation, enviro_data=enviro_dict, lat=parsed_lat, lon=parsed_lon)
        print(f"\nTotal Yield: {result['TotalYield']} kg/ha")
        print(f"\nYearly Yields:")
        for year, yield_val in enumerate(result['YearlyYields'], 1):
            print(f"  Year {year}: {yield_val} kg/ha")
        print()
    except Exception as e:
        print(f"Simulation Failed: {e}\n")