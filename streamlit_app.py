import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# KINETICGRID
# ============================================================

st.set_page_config(
    page_title="KineticGrid",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_ENERGY_PER_STEP_MJ = 0.5
DEFAULT_EFFECTIVE_PERCENT = 30
DEFAULT_STORAGE_EFFICIENCY = 80


# ============================================================
# PROJECT DIRECTORY
# Works locally, in Colab, and on Streamlit Cloud
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "location_model.pkl"
ENCODER_PATH = BASE_DIR / "location_encoder.pkl"
LOCATIONS_PATH = BASE_DIR / "locations.csv"


# ============================================================
# LOAD KINETICGRID COMPONENTS
# ============================================================

@st.cache_resource
def load_kineticgrid():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "location_model.pkl was not found."
        )

    if not ENCODER_PATH.exists():
        raise FileNotFoundError(
            "location_encoder.pkl was not found."
        )

    if not LOCATIONS_PATH.exists():
        raise FileNotFoundError(
            "locations.csv was not found."
        )

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    locations = pd.read_csv(LOCATIONS_PATH)

    return model, encoder, locations


# ============================================================
# STARTUP
# ============================================================

try:

    model, encoder, locations = load_kineticgrid()

except Exception as e:

    st.error("⚠️ KineticGrid could not start.")
    st.error(str(e))
    st.stop()


# ============================================================
# VALIDATE LOCATION FILE
# ============================================================

required_location_columns = [
    "Sensor_ID",
    "Sensor_Name"
]

missing_columns = [
    column
    for column in required_location_columns
    if column not in locations.columns
]

if missing_columns:

    st.error(
        "❌ locations.csv is missing required columns:"
    )

    st.write(missing_columns)

    st.stop()


available_locations = sorted(
    locations["Sensor_Name"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


if not available_locations:

    st.error(
        "❌ No locations were found in locations.csv."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("⚡ KINETICGRID")

st.subheader(
    "AI-Assisted Footstep Energy-Harvesting Platform"
)

st.write(
    "KineticGrid connects AI-based pedestrian prediction "
    "with footstep estimation, energy generation simulation, "
    "storage estimation, and physical prototype calibration."
)

st.divider()


# ============================================================
# MAIN DEPLOYMENT ENGINE
# ============================================================

st.header("🌐 KineticGrid Deployment Engine")

st.write(
    "Select the deployment location and operating conditions "
    "to run the KineticGrid prediction."
)


# ============================================================
# LOCATION
# ============================================================

location = st.selectbox(
    "📍 Select Location",
    available_locations
)


# ============================================================
# TIME PARAMETERS
# ============================================================

c1, c2, c3 = st.columns(3)

with c1:

    hour = st.slider(
        "🕐 Hour",
        min_value=0,
        max_value=23,
        value=17
    )

with c2:

    day_of_week = st.slider(
        "📅 Day of Week",
        min_value=0,
        max_value=6,
        value=4,
        help="0 = Monday, 6 = Sunday"
    )

with c3:

    month = st.slider(
        "📆 Month",
        min_value=1,
        max_value=12,
        value=8
    )


# ============================================================
# ENERGY PARAMETERS
# ============================================================

st.subheader("⚙️ Energy System Parameters")

e1, e2, e3 = st.columns(3)

with e1:

    energy_per_step = st.number_input(
        "Energy per Effective Step (mJ)",
        min_value=0.0,
        value=DEFAULT_ENERGY_PER_STEP_MJ,
        step=0.1
    )

with e2:

    effective_percent = st.slider(
        "Effective Footsteps (%)",
        min_value=0,
        max_value=100,
        value=DEFAULT_EFFECTIVE_PERCENT
    )

with e3:

    storage_efficiency = st.slider(
        "Storage Efficiency (%)",
        min_value=0,
        max_value=100,
        value=DEFAULT_STORAGE_EFFICIENCY
    )


# ============================================================
# RUN KINETICGRID
# ============================================================

run_prediction = st.button(
    "⚡ RUN KINETICGRID",
    type="primary",
    use_container_width=True
)


if run_prediction:

    try:

        # ====================================================
        # FIND SELECTED SENSOR
        # ====================================================

        selected = locations[
            locations["Sensor_Name"].astype(str) == location
        ]

        if selected.empty:

            st.error(
                "❌ Selected location could not be found."
            )

            st.stop()


        sensor_id = selected.iloc[0]["Sensor_ID"]


        # ====================================================
        # ENCODE SENSOR
        # ====================================================

        sensor_code = encoder.transform(
            [str(sensor_id)]
        )[0]


        # ====================================================
        # MODEL INPUT
        #
        # These are the features used by the existing
        # KineticGrid model.
        # ====================================================

        model_input = pd.DataFrame({

            "Sensor_Code": [
                sensor_code
            ],

            "Hour": [
                hour
            ],

            "DayOfWeek": [
                day_of_week
            ],

            "MonthNum": [
                month
            ],

            "DayOfMonth": [
                1
            ],

            "IsWeekend": [
                int(day_of_week >= 5)
            ]

        })


        # ====================================================
        # AI PREDICTION
        # ====================================================

        predicted_footfall = float(
            model.predict(model_input)[0]
        )

        predicted_footfall = max(
            0.0,
            predicted_footfall
        )


        # ====================================================
        # EFFECTIVE FOOTSTEPS
        # ====================================================

        effective_steps = (
            predicted_footfall
            * effective_percent
            / 100
        )


        # ====================================================
        # ENERGY GENERATION
        # ====================================================

        generated_mj = (
            effective_steps
            * energy_per_step
        )


        # ====================================================
        # USABLE ENERGY AFTER STORAGE EFFICIENCY
        # ====================================================

        usable_mj = (
            generated_mj
            * storage_efficiency
            / 100
        )


        # ====================================================
        # RESULTS
        # ====================================================

        st.divider()

        st.header("🧠 AI Prediction")


        r1, r2, r3 = st.columns(3)


        with r1:

            st.metric(
                "👣 Predicted Footfall",
                f"{predicted_footfall:,.0f}/hour"
            )


        with r2:

            st.metric(
                "👣 Effective Footsteps",
                f"{effective_steps:,.0f}"
            )


        with r3:

            if predicted_footfall >= 1000:

                rating = "HIGH 🟢"

            elif predicted_footfall >= 500:

                rating = "MODERATE 🟡"

            else:

                rating = "LOW 🔵"


            st.metric(
                "Deployment Potential",
                rating
            )


        # ====================================================
        # ENERGY RESULTS
        # ====================================================

        st.divider()

        st.header("⚡ Energy Generation")


        a1, a2, a3 = st.columns(3)


        with a1:

            st.metric(
                "Energy / Effective Step",
                f"{energy_per_step:.3f} mJ"
            )


        with a2:

            st.metric(
                "Generated Energy",
                f"{generated_mj:.3f} mJ"
            )


        with a3:

            st.metric(
                "Usable Energy",
                f"{usable_mj:.3f} mJ"
            )


        # ====================================================
        # PIPELINE
        # ====================================================

        st.divider()

        st.header("🔄 KineticGrid Energy Pipeline")


        st.info(
            f"""
📍 **Location:** {location}

🧠 **AI Prediction:**  
{predicted_footfall:,.0f} pedestrians/hour

↓

👣 **Effective Footsteps:**  
{effective_steps:,.0f}

↓

⚡ **Generated Energy:**  
{generated_mj:.3f} mJ

↓

🔋 **Usable Stored Energy:**  
{usable_mj:.3f} mJ
"""
        )


        st.success(
            "🟢 KineticGrid prediction and energy simulation completed."
        )


    except Exception as e:

        st.error(
            f"❌ Prediction failed: {e}"
        )


# ============================================================
# MANUAL ENERGY SIMULATION
# ============================================================

st.divider()

st.header("🔬 Manual Energy Simulation")

st.write(
    "Test the energy-harvesting system independently "
    "from the AI prediction."
)


s1, s2 = st.columns(2)


with s1:

    manual_steps = st.number_input(
        "👣 Number of Footsteps",
        min_value=0,
        value=100,
        step=1
    )


with s2:

    manual_energy = st.number_input(
        "⚡ Energy per Step (mJ)",
        min_value=0.0,
        value=DEFAULT_ENERGY_PER_STEP_MJ,
        step=0.1
    )


run_simulation = st.button(
    "🔬 RUN ENERGY SIMULATION",
    use_container_width=True
)


if run_simulation:

    total_energy_mj = (
        manual_steps
        * manual_energy
    )


    usable_energy_mj = (
        total_energy_mj
        * storage_efficiency
        / 100
    )


    x1, x2 = st.columns(2)


    with x1:

        st.metric(
            "⚡ Generated Energy",
            f"{total_energy_mj:.3f} mJ"
        )


    with x2:

        st.metric(
            "🔋 Usable Energy",
            f"{usable_energy_mj:.3f} mJ"
        )


# ============================================================
# PHYSICAL PROTOTYPE MODE
# ============================================================

st.divider()

st.header("🔧 Physical Prototype Mode")

st.write(
    "When the physical prototype is ready, enter the "
    "measured electrical readings here. The simulation "
    "parameters can then be replaced with real prototype data."
)


p1, p2, p3, p4 = st.columns(4)


with p1:

    voltage = st.number_input(
        "Voltage (V)",
        min_value=0.0,
        value=0.0,
        step=0.01
    )


with p2:

    current = st.number_input(
        "Current (A)",
        min_value=0.0,
        value=0.0,
        step=0.001
    )


with p3:

    duration = st.number_input(
        "Measurement Duration (s)",
        min_value=0.0,
        value=1.0,
        step=0.1
    )


with p4:

    prototype_steps = st.number_input(
        "Actual Footsteps",
        min_value=1,
        value=1,
        step=1
    )


process_prototype = st.button(
    "🔧 PROCESS PROTOTYPE READING",
    use_container_width=True
)


if process_prototype:

    # ========================================================
    # ELECTRICAL ENERGY
    # E = V × I × t
    # ========================================================

    energy_j = (
        voltage
        * current
        * duration
    )


    energy_mj = (
        energy_j
        * 1000
    )


    measured_per_step_mj = (
        energy_mj
        / prototype_steps
    )


    # ========================================================
    # RESULTS
    # ========================================================

    st.subheader("🧪 Prototype Measurement Result")


    q1, q2, q3 = st.columns(3)


    with q1:

        st.metric(
            "Measured Energy",
            f"{energy_j:.6f} J"
        )


    with q2:

        st.metric(
            "Measured Energy",
            f"{energy_mj:.3f} mJ"
        )


    with q3:

        st.metric(
            "Measured Energy / Step",
            f"{measured_per_step_mj:.6f} mJ"
        )


    st.success(
        "🟢 Prototype reading processed successfully."
    )


# ============================================================
# PROTOTYPE INTEGRATION NOTE
# ============================================================

st.divider()

st.subheader("🔌 Future Hardware Integration")

st.write(
    """
The current prototype section is designed so that the
theoretical energy-per-step value can eventually be replaced
with experimentally measured values from the physical
KineticGrid prototype.

Once hardware testing begins, measured voltage, current,
duration and actual footsteps can be entered directly into
this interface.
"""
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "⚡ KineticGrid — AI → Footsteps → Energy → Storage → Prototype"
)
