# ⚡ KineticGrid

### AI-Assisted Footstep Energy-Harvesting Platform

KineticGrid is an AI-assisted energy-harvesting platform designed to estimate pedestrian activity, simulate potential footstep-based energy generation, and provide a software interface for future integration with a physical energy-harvesting prototype.

The system connects **AI prediction, footstep estimation, energy simulation, and physical prototype calibration** into one workflow.

---

## 🚀 Core Concept

KineticGrid follows a simple pipeline:

**📍 Location → 🧠 AI Prediction → 👣 Footsteps → ⚡ Energy Generation → 🔋 Storage → 🔧 Prototype**

The software can estimate pedestrian activity for a selected location and convert the predicted activity into an estimated amount of recoverable energy.

Once the physical prototype is available, real electrical measurements can be entered into the application to replace the temporary simulation assumptions.

---

## ✨ Features

### 🧠 AI Footfall Prediction

Predicts pedestrian activity using the trained KineticGrid machine-learning model based on:

* Location
* Hour
* Day of week
* Month
* Weekend/weekday conditions

### 👣 Footstep Estimation

Converts predicted pedestrian activity into an estimated number of effective footsteps interacting with the energy-harvesting system.

### ⚡ Energy Simulation

Uses configurable energy-per-step and system-efficiency parameters to estimate theoretical energy generation.

### 🔋 Storage Estimation

Estimates usable energy after applying storage/system efficiency.

### 🔬 Manual Simulation

Allows users to directly enter a number of footsteps and energy-per-step value to test the energy-harvesting concept independently of the AI model.

### 🔧 Physical Prototype Mode

Designed for future hardware integration.

Users can enter:

* Measured voltage
* Measured current
* Measurement duration
* Actual number of footsteps

The application calculates the measured energy and energy generated per footstep.

---

## 🏗️ System Architecture

```text
                KINETICGRID
                     │
                     ▼
              Select Location
                     │
                     ▼
              🧠 ML Prediction
                     │
                     ▼
             Predicted Footfall
                     │
                     ▼
              👣 Effective Steps
                     │
                     ▼
             ⚡ Energy Estimation
                     │
                     ▼
               🔋 Storage
                     │
                     ▼
          🔧 Physical Prototype
                     │
                     ▼
             Real Measurements
```

---

## 🖥️ Application

The current application is built using **Streamlit**.

Main application file:

```text
streamlit_app.py
```

---

## 📁 Project Structure

```text
KineticGrid/
│
├── streamlit_app.py
├── requirements.txt
├── location_encoder.pkl
├── locations.csv
│
└── README.md
```

The trained ML model is hosted separately because of its large file size.

---

## ⚙️ Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run streamlit_app.py
```

The application will open in a browser.

---

## 🔬 Prototype Calibration

The current energy-per-step value is a configurable simulation parameter.

When the physical KineticGrid prototype is available, measured electrical output can be entered directly into the application.

For a measured voltage `V`, current `I`, and duration `t`:

```text
Energy = Voltage × Current × Time
```

The measured energy can then be divided by the actual number of footsteps to obtain an experimental energy-per-step value.

This allows the software model to progressively move from **theoretical simulation** toward **real prototype measurements**.

---

## 🎯 Development Status

| Component                       | Status         |
| ------------------------------- | -------------- |
| AI pedestrian prediction        | ✅ Working      |
| Location-based prediction       | ✅ Working      |
| Footstep estimation             | ✅ Working      |
| Energy simulation               | ✅ Working      |
| Manual simulation               | ✅ Working      |
| Prototype measurement interface | ✅ Working      |
| Physical hardware integration   | 🔧 Future      |
| Experimental calibration        | 🔧 Future      |
| Full deployment optimization    | 🔧 In progress |

---

## 🌱 Future Development

Future versions of KineticGrid can integrate:

* Real-time sensor readings
* Piezoelectric/energy-harvesting hardware
* Battery or supercapacitor monitoring
* Real-time energy dashboards
* Automatic prototype calibration
* IoT-based monitoring
* Multiple deployed locations
* Improved AI prediction models

---

## 💡 Vision

KineticGrid aims to explore how everyday human movement can become a source of useful distributed energy.

Instead of treating pedestrian traffic only as movement data, KineticGrid attempts to connect **where people move → how many footsteps occur → how much energy could be harvested → how much energy the physical system actually produces.**

---

## ⚡ KineticGrid

**AI → Footsteps → Energy → Storage → Real-World Prototype**

Built as an experimental technology platform for AI-assisted kinetic energy harvesting.
