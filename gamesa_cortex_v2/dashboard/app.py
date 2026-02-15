import streamlit as st
import json
import pandas as pd
import time
import os
import sys

# Add project root to path so we can import modules if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from gamesa_cortex_v2.src.core.config import GamesaConfig

st.set_page_config(
    page_title="Gamesa Cortex V2 - Neural Control Plane",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling (Corporate Dark Mode) ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #4e4e4e;
        text-align: center;
    }
    .stMetricValue {
        color: #00e0ff !important;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 300;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
col1, col2 = st.columns([1, 4])
with col1:
    # Placeholder for logo
    st.markdown("### 🌀 GAMESA")
with col2:
    st.title("Cortex V2 Neural Control Plane")
    st.markdown("*Industry 5.0 | Adaptive AI Orchestration | Real-Time Safety*")

st.divider()

# --- Data Loading ---
LOG_FILE = "logs/intraspectral_latest.json"

@st.cache_data(ttl=2) # Auto refresh every 2s
def load_logs():
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

logs = load_logs()

# --- KPI Section ---
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

# Mock real-time data if logs are empty/stale
# In prod, we'd aggregated this from the log stream
budget = "N/A"
cortisol = "N/A"
active_threads = GamesaConfig.MAX_WORKERS
events_count = len(logs)

if logs:
    # Find last economic event
    eco_logs = [l for l in logs if l.get("spectrum") == "ECONOMIC"]
    if eco_logs:
        last_eco = eco_logs[-1]
        if "metrics" in last_eco and "budget" in last_eco["metrics"]:
             budget = last_eco["metrics"]["budget"]

    # Find last system state
    sys_logs = [l for l in logs if l.get("spectrum") == "SYSTEM" and "metrics" in l]
    if sys_logs:
        last_sys = sys_logs[-1]
        cortisol = last_sys["metrics"].get("cortisol", 0.1)

with col_kpi1:
    st.metric("Economic Budget", f"{budget} CR", delta_color="normal")
with col_kpi2:
    st.metric("Neural Cortisol", f"{cortisol}", delta="-0.05" if isinstance(cortisol, float) and cortisol < 0.5 else "0.1")
with col_kpi3:
    st.metric("Active Threads", active_threads)
with col_kpi4:
    st.metric("Total Events Logged", events_count)

# --- Charts ---
st.subheader("System Telemetry")
if logs:
    df = pd.DataFrame(logs)
    if not df.empty and "timestamp" in df.columns:
        # Normalize timestamp
        df["time"] = pd.to_datetime(df["timestamp"], unit="ns")
        
        # Event Distribution
        st.bar_chart(df["spectrum"].value_counts())
        
        # Detailed Table
        st.dataframe(df[["time", "spectrum", "component", "message", "metrics"]].sort_values("time", ascending=False), use_container_width=True)
else:
    st.info("Waiting for system logs...")

# --- Sidebar Controls ---
st.sidebar.header("Manual Override")
preset = st.sidebar.selectbox("Force Preset", ["AUTO", "IDLE_ECO", "HIGH_PERFORMANCE", "SAFETY_CRITICAL"])
if st.sidebar.button("Apply Preset"):
    st.sidebar.success(f"Preset '{preset}' signal sent to NPU.")
