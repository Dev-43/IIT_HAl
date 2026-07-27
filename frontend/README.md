# AeroOptima Frontend — Next.js Sizing & SImulation Dashboard

This directory houses the interactive React dashboard for visualization and control of the hybrid UAV optimization platform.

---

## 🎨 User Interface Features

The interface is structured as a single-page responsive dashboard divided into three core control and telemetry areas:

### 1. Control & Sizing Panel (Left)
* **Mission Scenarios:** Configurable sliders for target cruise speed ($150\text{--}350\text{ km/h}$), cruise altitude ($3000\text{--}10000\text{ m}$), payload weight ($100\text{--}300\text{ kg}$), and initial fuel fraction ($10\%\text{--}100\%$).
* **Heuristic Profile Timeline:** Displays the duration of each flight phase calculated by the simulation.
* **MTOW Weight Breakdown:** An interactive bar chart visualizing how the Maximum Takeoff Weight (1000 kg) is shared among the structural airframe, fuel payload, and propulsion components (engine, battery, and motor).
* **GA Execution Controls:** Buttons to trigger the genetic optimization loop on the backend.

### 2. Tabbed Visualizer Area (Center)
* **3D Flight Profile:** Powered by **Three.js** and `@react-three/fiber`/`drei`. Renders a low-poly terrain grid and a 3D tactical flight path curve. 
  * The trail is **color-coded** by the active power source: **Amber** (Battery only), **Cyan** (Hybrid parallel mode), **Green** (Engine only), and **Gray** (Gliding/Idle).
  * Features waypoint markers, a 3D coordinate compass, interactive orbit controls, a playback scrubber, and an active HUD floating above the UAV.
* **Telemetry Charts:** Dynamic plots rendered by **Plotly.js** mapping:
  1. *Power Split:* Power required vs engine output vs electric motor output. Correctly displays actual power sharing on unified hover.
  2. *Resource Status:* Battery SoC (%) and fuel remaining (kg) depletion curves.
  3. *Trajectory:* Flight speed (m/s) and altitude (m) profile curves.

### 3. Dense Telemetry Matrix (Right - Collapsible)
* A high-density table displaying scrollable flight stats updated every time step.
* Column headers: `Time (hh:mm:ss)`, `P_aero (kW)`, `P_climb (kW)`, `P_req (kW)`, `PSR (%)`, `Motor (kW)`, `Engine (kW)`, `SoC (%)`, `Fuel (kg)`, and `Phase`.
* Auto-syncs highlight state smoothly with the 3D playback scrubber to prevent flickering.

---

## 🛠️ Code Structure

* **`src/app/page.tsx`:** Primary page routing layout.
* **`src/components/Dashboard.tsx`:** Manages state (loading, telemetry, current scrubber index), API fetches, and layouts.
* **`src/components/FlightScene.tsx`:** Handles the Three.js canvas, generating the 3D racetrack orbit path, camera angles, waypoints, and the UAV HUD.
* **`src/components/TelemetryChart.tsx`:** Configures and renders the Plotly.js charts with unified hover tooltip formatters.
* **`src/components/TelemetryTable.tsx`:** Manages the downsampled scrollable tabular grid, matching current scrubber state and formatting.

---

## 🚀 Commands

### Run Development Server
```bash
npm run dev
```

### Build Production Bundle
```bash
npm run build
```

### Start Production Server
```bash
npm run start
```
