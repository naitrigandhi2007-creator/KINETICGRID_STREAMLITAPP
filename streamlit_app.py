%%writefile streamlit_app.py

import os
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
# PARAMETERS
# ============================================================

DEFAULT_ENERGY_PER_STEP_MJ = 0.5
DEFAULT_EFFECTIVE_PERCENT = 30
DEFAULT_STORAGE_EFFICIENCY = 80


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_kineticgrid():

    model_path = "/content/location_model.pkl"
    encoder_path = "/content/location_encoder.pkl"
    locations_path = "/content/locations.csv"

    if not os.path.exists(model_path):
        raise FileNotFoundError("location_model.pkl is missing.")

    if not os.path.exists(encoder_path):
        raise FileNotFoundError("location_encoder.pkl is missing.")

    if not os.path.exists(locations_path):
        raise FileNotFoundError("locations.csv is missing.")

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    locations = pd.read_csv(locations_path)

    return model, encoder, locations


# ============================================================
# STARTUP
# ============================================================

try:

    model, encoder, locations = load_kineticgrid()

except Exception as e:

    st.error(f"❌ KineticGrid startup failed: {e}")
    st.stop()


# ============================================================
# LOCATION COLUMN
# ============================================================

if "Sensor_Name" not in locations.columns:

    st.error(
        "❌ locations.csv does not contain the Sensor_Name column."
    )

    st.write("Columns found:", locations.columns.tolist())

    st.stop()


available_locations = sorted(
    locations["Sensor_Name"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


# ============================================================
# HEADER
# ============================================================

st.title("⚡ KINETICGRID")

st.subheader(
    "AI-Assisted Footstep Energy-Harvesting Platform"
)

st.write(
    "Predict pedestrian activity, estimate effective footsteps, "
    "simulate energy generation, and prepare the system for "
    "physical prototype calibration."
)

st.divider()


# ============================================================
# DEPLOYMENT ENGINE
# ============================================================

st.header("🌐 KineticGrid Deployment Engine")

location = st.selectbox(
    "📍 Select Location",
    available_locations
)

col1, col2, col3 = st.columns(3)

with col1:

    hour = st.slider(
        "🕐 Hour",
        0,
        23,
        17
    )

with col2:

    day_of_week = st.slider(
        "📅 Day of Week",
        0,
        6,
        4,
        help="0 = Monday, 6 = Sunday"
    )

with col3:

    month = st.slider(
        "📆 Month",
        1,
        12,
        8
    )


# ============================================================
# ENERGY PARAMETERS
# ============================================================

st.subheader("⚙️ Energy Parameters")

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
        0,
        100,
        DEFAULT_EFFECTIVE_PERCENT
    )

with e3:

    storage_efficiency = st.slider(
        "Storage Efficiency (%)",
        0,
        100,
        DEFAULT_STORAGE_EFFICIENCY
    )


# ============================================================
# RUN
# ============================================================

run = st.button(
    "⚡ RUN KINETICGRID",
    type="primary",
    use_container_width=True
)


if run:

    try:

        # ----------------------------------------------------
        # FIND SENSOR
        # ----------------------------------------------------

        selected = locations[
            locations["Sensor_Name"].astype(str) == location
        ]

        if selected.empty:

            st.error("Location not found.")

            st.stop()


        sensor_id = selected.iloc[0]["Sensor_ID"]


        # ----------------------------------------------------
        # ENCODE SENSOR
        # ----------------------------------------------------

        sensor_code = encoder.transform(
            [str(sensor_id)]
        )[0]


        # ----------------------------------------------------
        # MODEL INPUT
        # ----------------------------------------------------

        model_input = pd.DataFrame({

            "Sensor_Code": [sensor_code],

            "Hour": [hour],

            "DayOfWeek": [day_of_week],

            "MonthNum": [month],

            "DayOfMonth": [1],

            "IsWeekend": [
                int(day_of_week >= 5)
            ]

        })


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        predicted_footfall = float(
            model.predict(model_input)[0]
        )

        predicted_footfall = max(
            0,
            predicted_footfall
        )


        # ----------------------------------------------------
        # EFFECTIVE STEPS
        # ----------------------------------------------------

        effective_steps = (
            predicted_footfall
            * effective_percent
            / 100
        )


        # ----------------------------------------------------
        # ENERGY
        # ----------------------------------------------------

        generated_mj = (
            effective_steps
            * energy_per_step
        )

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
        # ENERGY
        # ====================================================

        st.divider()

        st.header("⚡ Energy Generation")

        a1, a2, a3 = st.columns(3)

        with a1:

            st.metric(
                "Energy / Step",
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


        st.success(
            "🟢 KineticGrid prediction and energy estimation completed."
        )


        # ====================================================
        # PIPELINE
        # ====================================================

        st.divider()

        st.header("🔄 KineticGrid Pipeline")

        st.write(
            f"""
📍 **{location}**

↓  

🧠 **AI Footfall Prediction:**  
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


    except Exception as e:

        st.error(
            f"❌ Prediction failed: {e}"
        )


# ============================================================
# MANUAL SIMULATION
# ============================================================

st.divider()

st.header("🔬 Manual Energy Simulation")

st.write(
    "Test the energy-harvesting concept independently "
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


if st.button(
    "🔬 RUN SIMULATION",
    use_container_width=True
):

    total_energy = (
        manual_steps
        * manual_energy
    )

    usable_energy = (
        total_energy
        * storage_efficiency
        / 100
    )

    x1, x2 = st.columns(2)

    with x1:

        st.metric(
            "Generated Energy",
            f"{total_energy:.3f} mJ"
        )

    with x2:

        st.metric(
            "Usable Energy",
            f"{usable_energy:.3f} mJ"
        )


# ============================================================
# PHYSICAL PROTOTYPE
# ============================================================

st.divider()

st.header("🔧 Physical Prototype Mode")

st.write(
    "When the physical prototype is available, enter "
    "measured voltage, current, duration and footsteps."
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
        "Duration (s)",
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


if st.button(
    "🔧 PROCESS PROTOTYPE READING",
    use_container_width=True
):

    energy_j = (
        voltage
        * current
        * duration
    )

    energy_mj = (
        energy_j
        * 1000
    )

    measured_per_step = (
        energy_mj
        / prototype_steps
    )


    st.subheader("🧪 Prototype Result")

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
            f"{measured_per_step:.6f} mJ"
        )


    st.success(
        "Prototype measurement processed successfully."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "⚡ KineticGrid — AI → Footsteps → Energy → Storage → Prototype"
)
