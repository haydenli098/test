import streamlit as st
import importlib.util
import time

# Import your existing modules
# We use aliases to avoid naming conflicts since they share function names
import MLP as mlp_backend
import RNN as rnn_backend

# Workaround to import a file with spaces in the name ('Rotation to yield.py')
spec = importlib.util.spec_from_file_location("rotation_to_yield", "Rotation to yield.py")
apsim_backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apsim_backend)

st.set_page_config(page_title="Crop Rotation Optimizer", layout="wide")

st.title("🌾 Crop Rotation Optimization Dashboard")
st.markdown("Use Digital Twins (MLP / RNN) or the APSIM Engine to find the optimal crop sequence for your paddock.")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Configuration")
engine_choice = st.sidebar.selectbox(
    "Select Prediction Engine:",
    ("MLP Surrogate", "RNN Surrogate", "APSIM Simulator")
)

st.sidebar.markdown("---")
st.sidebar.header("Location Input")
lat_input = st.sidebar.number_input("Latitude", value=-33.8, format="%.4f")
lon_input = st.sidebar.number_input("Longitude", value=151.2, format="%.4f")

if engine_choice == "APSIM Simulator":
    rotation_to_test = st.sidebar.text_input("Crop Sequence to Simulate", value="Wheat, Canola, Wheat, Chickpea, Barley")

# Caching the training step so it only happens ONCE while the app is running
@st.cache_resource
def load_mlp():
    st.toast("Loading MLP Model... Please wait.", icon="⏳")
    return mlp_backend.load_or_train_model()

@st.cache_resource
def load_rnn():
    st.toast("Loading RNN Model... Please wait.", icon="⏳")
    return rnn_backend.load_or_train_model()

# --- MAIN DASHBOARD ---
st.write(f"### Currently using: **{engine_choice}**")

if st.sidebar.button("Run Optimizer / Simulator", type="primary"):
    
    with st.spinner("Processing Data..."):
        try:
            if engine_choice == "MLP Surrogate":
                # 1. Load/Train Model
                trained_model, scaler_x, scaler_y = load_mlp()
                
                # 2. Fetch Enviro Data
                enviro_data = mlp_backend.get_enviro_data(lat_input, lon_input)
                st.info(f"📍 Matched Location Data: Rainfall={enviro_data[0]:.1f}mm, Elevation={enviro_data[1]:.1f}m")
                
                # 3. Optimize
                best_seq, expected_revenue, best_yields = mlp_backend.optimize_rotation(
                    trained_model, scaler_x, scaler_y, enviro_data, iterations=2000
                )
                
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
                trained_model, scaler_x, scaler_y = load_rnn()
                
                # 2. Fetch Enviro Data
                enviro_data = rnn_backend.get_enviro_data(lat_input, lon_input)
                st.info(f"📍 Matched Location Data: Rainfall={enviro_data[0]:.1f}mm, N={enviro_data[6]:.1f}kg/ha")
                
                # 3. Optimize
                best_seq, expected_revenue, best_yields = rnn_backend.optimize_rotation(
                    trained_model, scaler_x, scaler_y, enviro_data, iterations=2000
                )
                
                st.success("Optimization Complete!")
                st.metric(label="Optimal 5-Year Rotation", value=best_seq)
                st.metric(label="Predicted Revenue", value=f"${expected_revenue:.2f} / ha")
                
                st.write("**Predicted Yearly Yields:**")
                cols = st.columns(5)
                crops = [c.strip() for c in best_seq.split(",")]
                for year, (crop, yield_val) in enumerate(zip(crops, best_yields)):
                    if year < 5:
                        cols[year].metric(label=f"Year {year+1} ({crop})", value=f"{yield_val:.1f} kg/ha")

            elif engine_choice == "APSIM Simulator":
                st.warning("Running the physical APSIM engine... This might take a moment.")
                
                st.write(f"**Simulating Rotation:** {rotation_to_test}")
                
                # 1. Run Simulation
                start_time = time.time()
                result = apsim_backend.simulate_rotation_yield(
                    rotation=rotation_to_test, 
                    lat=lat_input, 
                    lon=lon_input, 
                    sim_id="streamlit_run"
                )
                
                st.success(f"Simulation Complete in {time.time() - start_time:.1f} seconds!")
                st.metric(label="Total 5-Year Yield", value=f"{result['TotalYield']} kg/ha")
                
                # Display Yearly Breakdown
                cols = st.columns(5)
                for year, yield_val in enumerate(result['YearlyYields']):
                    cols[year].metric(label=f"Year {year+1}", value=f"{yield_val}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
