import json
import csv
import re
import os
import glob

# Replace with your actual folder path
input_folder = r'c:\Users\flami\Documents\VSCode\ENGG2112\8307d1ca-4fb0-4322-a182-89849d355b24'
output_file = 'ansis_mapped_sites_combined.csv'

def extract_coords(geometry_str):
    # Extract numbers from "SRID=4283;POINT(153.073 -29.333)"
    match = re.search(r'POINT\(([-\d.]+) ([-\d.]+)\)', geometry_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

# Header for the CSV
csv_header = ['ID', 'Longitude', 'Latitude', 'Elevation_m', 'SoilDepth_m', 'Drainage', 'Permeability', 'pH', 'OrganicCarbon_pct']

# Find all files in the folder (since the files lack a .json extension)
json_files = [f for f in glob.glob(os.path.join(input_folder, '*')) if os.path.isfile(f)]

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    
    for input_file in json_files:
        try:
            with open(input_file, 'r') as json_f:
                ansis_data = json.load(json_f)
            
            sites = ansis_data.get('data', [])
            
            for site in sites:
                site_id = site.get('scopedIdentifier', [{}])[0].get('value', 'Unknown')
                geom_str = site.get('geometry', {}).get('result', '')
                lon, lat = extract_coords(geom_str)
                
                # Extract Elevation
                elev = site.get('elevation', [{}])[0].get('result', {}).get('value', 'N/A')
                
                # Extract soil properties
                soil_depth = 'N/A'
                drainage = 'N/A'
                permeability = 'N/A'
                ph = 'N/A'
                organic_carbon = 'N/A'
                
                site_visits = site.get('siteVisit', [])
                if site_visits:
                    soil_profile = site_visits[0].get('soilProfile', [])
                    if soil_profile:
                        profile = soil_profile[0]
                        soil_depth = profile.get('depth', {}).get('result', {}).get('value', 'N/A')
                        drainage_raw = profile.get('drainage', {}).get('result', 'N/A')
                        if drainage_raw != 'N/A':
                            parts = drainage_raw.split('-')
                            if parts and parts[-1].isdigit():
                                drainage = parts[-1]
                        permeability_raw = profile.get('permeability', {}).get('result', 'N/A')
                        if permeability_raw != 'N/A':
                            parts = permeability_raw.split('-')
                            if parts and parts[-1].isdigit():
                                permeability = parts[-1]
                        
                        soil_layers = profile.get('soilLayer', [])
                        if soil_layers:
                            layer = soil_layers[0]  # Top layer
                            ph_values = layer.get('ph', [])
                            if ph_values:
                                ph = ph_values[0].get('result', {}).get('value', 'N/A')
                            organic_carbons = layer.get('organicCarbon', [])
                            if organic_carbons:
                                organic_carbon = organic_carbons[0].get('result', {}).get('value', 'N/A')
                            elif layer.get('totalOrganicCarbon'):
                                organic_carbons = layer.get('totalOrganicCarbon', [])
                                if organic_carbons:
                                    organic_carbon = organic_carbons[0].get('result', {}).get('value', 'N/A')
                
                if lon and lat:
                    writer.writerow([site_id, lon, lat, elev, soil_depth, drainage, permeability, ph, organic_carbon])
            
            print(f"Processed: {os.path.basename(input_file)}")
        except Exception as e:
            print(f"Error processing {input_file}: {e}")

print(f"Conversion complete! Processed {len(json_files)} files into {output_file}")