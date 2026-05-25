import streamlit as st

# Import your existing modules
# We use aliases to avoid naming conflicts since they share function names
import MLP as mlp_backend
import RNN as rnn_backend

# NB requires loading data and training - do this lazily to avoid timeout
try:
    import NB as nb_backend
except Exception as e:
    nb_backend = None
    st.warning(f"⚠️ Naive Bayes unavailable: {str(e)[:100]}")

st.set_page_config(page_title="Crop Rotation Optimizer", layout="wide")

st.title("🌾 Crop Rotation Optimization Dashboard")
st.markdown("Use Digital Twins (MLP / RNN / Naive Bayes) to find the optimal crop sequence for your paddock.")

# --- STARTUP CHECKS ---
import os

required_files = {
    'final_dataset.csv': 'Location/environment data',
    'final_crop_rotation_plan.csv': 'Crop rotation training data',
}

missing_files = [f for f in required_files.keys() if not os.path.exists(f)]
if missing_files:
    st.error(f"❌ Missing required data files: {', '.join(missing_files)}")
    st.info("Please ensure these CSV files are in the same directory as this app.")
    st.stop()

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Configuration")
engine_options = ["MLP Surrogate", "RNN Surrogate"]
if nb_backend is not None:
    engine_options.append("Naive Bayes Classifier")

engine_choice = st.sidebar.selectbox(
    "Select Prediction Engine:",
    engine_options
)

st.sidebar.markdown("---")
st.sidebar.header("Location Input")
lat_input = st.sidebar.number_input("Latitude", value=-33.8, format="%.4f")
lon_input = st.sidebar.number_input("Longitude", value=151.2, format="%.4f")

# Caching the training step so it only happens ONCE while the app is running
@st.cache_resource
def load_mlp():
    return mlp_backend.load_or_train_model()

@st.cache_resource
def load_rnn():
    return rnn_backend.load_or_train_model()

@st.cache_resource
def load_nb():
    # Load NB model (trains on startup)
    return nb_backend

# --- MAIN DASHBOARD ---
st.write(f"### Currently using: **{engine_choice}**")

if st.sidebar.button("Run Optimizer / Simulator", type="primary"):
    
    with st.spinner("Processing Data..."):
            if engine_choice == "MLP Surrogate":
                # 1. Load/Train Model
                st.toast("Loading MLP Model... Please wait.", icon="⏳")
                try:
                    trained_model, scaler_x, scaler_y = load_mlp()
                except Exception as e:
                    st.error(f"Error loading MLP model: {e}")
                    st.stop()
                
                # 2. Fetch Enviro Data
                try:
                    enviro_data = mlp_backend.get_enviro_data(lat_input, lon_input)
                except Exception as e:
                    st.error(f"Error fetching environment data: {e}")
                    st.stop()
                    
                st.info(f"📍 Matched Location Data: Rainfall={enviro_data[0]:.1f}mm, Elevation={enviro_data[1]:.1f}m")
                
                # 3. Optimize
                try:
                    best_seq, expected_revenue, best_yields = mlp_backend.optimize_rotation(
                        trained_model, scaler_x, scaler_y, enviro_data, iterations=2000
                    )
                except Exception as e:
                    st.error(f"Error optimizing rotation: {e}")
                    st.stop()
                
                st.success("Optimization Complete!")
                st.metric(label="Optimal 5-Year Rotation", value=best_seq)
                st.metric(label="Predicted Revenue", value=f"${expected_revenue:.2f} / ha")
                
                st.write("**Predicted Yearly Yields:**")
                cols = st.columns(5)
                crops = [c.strip() for c in best_seq.split(",")]
                for year, (crop, yield_val) in enumerate(zip(crops, best_yields)):
                    if year < 5:
                        cols[year].metric(label=f"Year {year+1} ({crop})", value=f"{yield_val:.1f} kg/ha")

            elif engine_choice == "RNN Surrogate":
                # 1. Load/Train Model
                st.toast("Loading RNN Model... Please wait.", icon="⏳")
                try:
                    trained_model, scaler_x, scaler_y = load_rnn()
                except Exception as e:
                    st.error(f"Error loading RNN model: {e}")
                    st.stop()
                
                # 2. Fetch Enviro Data
                try:
                    enviro_data = rnn_backend.get_enviro_data(lat_input, lon_input)
                except Exception as e:
                    st.error(f"Error fetching environment data: {e}")
                    st.stop()
                    
                st.info(f"📍 Matched Location Data: Rainfall={enviro_data[0]:.1f}mm, N={enviro_data[6]:.1f}kg/ha")
                
                # 3. Optimize
                try:
                    best_seq, expected_revenue, best_yields = rnn_backend.optimize_rotation(
                        trained_model, scaler_x, scaler_y, enviro_data, iterations=2000
                    )
                except Exception as e:
                    st.error(f"Error optimizing rotation: {e}")
                    st.stop()
                
                st.success("Optimization Complete!")
                st.metric(label="Optimal 5-Year Rotation", value=best_seq)
                st.metric(label="Predicted Revenue", value=f"${expected_revenue:.2f} / ha")
                
                st.write("**Predicted Yearly Yields:**")
                cols = st.columns(5)
                crops = [c.strip() for c in best_seq.split(",")]
                for year, (crop, yield_val) in enumerate(zip(crops, best_yields)):
                    if year < 5:
                        cols[year].metric(label=f"Year {year+1} ({crop})", value=f"{yield_val:.1f} kg/ha")

            elif engine_choice == "Naive Bayes Classifier":
                if nb_backend is None:
                    st.error("❌ Naive Bayes model unavailable.")
                    st.stop()
                
                # 1. Load NB Model
                st.toast("Loading Naive Bayes Model... Please wait.", icon="⏳")
                try:
                    nb_model = load_nb()
                except Exception as e:
                    st.error(f"Error loading Naive Bayes model: {e}")
                    st.stop()
                
                # 2. Predict rotation using coordinates
                st.info(f"📍 Finding nearby location with similar soil properties...")
                try:
                    predicted_rotation = nb_model.predict_crop_rotation_by_coordinates(lat_input, lon_input)
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    st.stop()
                
                st.success("Prediction Complete!")
                st.metric(label="Recommended Crop Rotation", value=predicted_rotation)
                st.write(f"*Based on soil properties and environmental conditions at ({lat_input:.4f}, {lon_input:.4f})*")
