# ============================================================
# MODEL FAILURE FORECASTER
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Model Failure Forecaster",
    page_icon="📉",
    layout="wide"
)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown("""
<h1 style='text-align:center;'>📉 MODEL FAILURE FORECASTER</h1>
<p style='text-align:center;color:#AAAAAA;'>
Interactive Dashboard for ML Model Health Monitoring
</p>
""", unsafe_allow_html=True)

st.divider()

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader("📂 Upload Dataset", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV file to begin")
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.strip()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if not numeric_cols:
    st.error("Numeric target column required")
    st.stop()

TARGET_COL = st.sidebar.selectbox("🎯 Target Column", numeric_cols)
TIME_COL = st.sidebar.selectbox("⏱️ Time Column (optional)", ["None"] + df.columns.tolist())
if TIME_COL == "None":
    TIME_COL = None

# ------------------------------------------------------------
# DATASET VIEW
# ------------------------------------------------------------
with st.expander("📂 Dataset Overview"):
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

# ------------------------------------------------------------
# TIME HANDLING
# ------------------------------------------------------------
if TIME_COL:
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).set_index(TIME_COL)
else:
    df["pseudo_time"] = range(len(df))
    df = df.set_index("pseudo_time")

df = df.drop(columns=["id"], errors="ignore").ffill()

# ------------------------------------------------------------
# SPLIT
# ------------------------------------------------------------
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

min_future = 20
split = max(len(df) - min_future, int(len(df) * 0.7))

X_train, X_future = X.iloc[:split], X.iloc[split:]
y_train, y_future = y.iloc[:split], y.iloc[split:]

if len(X_future) < 5:
    st.error("Not enough future data to compute Health Trend.")
    st.stop()

X_train = pd.get_dummies(X_train, drop_first=True)
X_future = pd.get_dummies(X_future, drop_first=True)
X_future = X_future.reindex(columns=X_train.columns, fill_value=0)

# ------------------------------------------------------------
# MODEL + MONITORING
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def run_monitoring(X_train, y_train, X_future, y_future):

    model = RandomForestRegressor(
        n_estimators=60,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_future)
    abs_error = np.abs(y_future.values - preds)
    mae = mean_absolute_error(y_future, preds)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_future_s = scaler.transform(X_future)

    train_mean = X_train_s.mean(axis=0)
    drift_per_sample = np.abs(X_future_s - train_mean).mean(axis=1)
    drift_score = drift_per_sample.mean()

    trees = model.estimators_[:20]
    tree_preds = np.array([t.predict(X_future) for t in trees])
    uncertainty_per_sample = tree_preds.std(axis=0)
    uncertainty_score = uncertainty_per_sample.mean()

    return abs_error, mae, drift_per_sample, drift_score, uncertainty_per_sample, uncertainty_score

with st.spinner("Running model monitoring..."):
    abs_error, mae, drift_per_sample, drift_score, uncertainty_per_sample, uncertainty_score = (
        run_monitoring(X_train, y_train, X_future, y_future)
    )

# ------------------------------------------------------------
# HEALTH SCORE
# ------------------------------------------------------------
window = max(3, int(len(abs_error) * 0.1))

def norm(x):
    return x / (np.nanpercentile(x, 95) + 1e-6)

health_series = 1 - (
    norm(pd.Series(abs_error).rolling(window, min_periods=1).mean()) +
    norm(pd.Series(drift_per_sample).rolling(window, min_periods=1).mean()) +
    norm(pd.Series(uncertainty_per_sample).rolling(window, min_periods=1).mean())
) / 3

health_series = (
    health_series
    .fillna(method="bfill")
    .fillna(method="ffill")
    .clip(0, 1)
)

health_pct = round(float(health_series.mean()) * 100, 1)

health_df = pd.DataFrame({"Health Score": health_series.values})

# ------------------------------------------------------------
# GAUGES
# ------------------------------------------------------------
st.subheader("🫀 System Vitals")

def gauge(title, value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%"},
        title={"text": title},
        gauge={"axis": {"range": [0, 100]}}
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

c1, c2, c3 = st.columns(3)
with c1: gauge("🧠 Model Health", health_pct)
with c2: gauge("🔀 Data Drift", min(drift_score * 100, 100))
with c3: gauge("❓ Uncertainty", min(uncertainty_score * 100, 100))

st.divider()

# ------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------
st.subheader("📊 Analysis View")

choice = st.radio(
    "Plots for the given datasets!!",
    ["Health Trend", "Prediction Error", "Data Drift", "Model Uncertainty"],
    horizontal=True
)

if choice == "Health Trend":
    st.line_chart(health_df, height=350)
elif choice == "Prediction Error":
    st.line_chart(pd.DataFrame({"Error": abs_error}))
elif choice == "Data Drift":
    st.line_chart(pd.DataFrame({"Drift": drift_per_sample}))
else:
    st.line_chart(pd.DataFrame({"Uncertainty": uncertainty_per_sample}))

# ------------------------------------------------------------
# FINAL MODEL VERDICT & INTERPRETATION
# ------------------------------------------------------------
st.divider()
st.subheader("🧠 Final Model Verdict & Interpretation")

# Model Health
if health_pct >= 75:
    health_state = "🟢 GOOD"
    health_msg = "Model performance is stable and reliable."
elif health_pct >= 50:
    health_state = "🟠 WARNING"
    health_msg = "Model performance is degrading. Retraining may be needed soon."
else:
    health_state = "🔴 CRITICAL"
    health_msg = "Model failure is likely. Immediate retraining required."

# Data Drift
if drift_score < 0.5:
    drift_state = "🟢 LOW"
    drift_msg = "Input data distribution is stable."
elif drift_score < 1.0:
    drift_state = "🟠 MODERATE"
    drift_msg = "Noticeable data drift detected. Monitor closely."
else:
    drift_state = "🔴 HIGH"
    drift_msg = "Severe data drift detected. Model assumptions may no longer hold."

# Uncertainty
if uncertainty_score < 0.3:
    unc_state = "🟢 LOW"
    unc_msg = "Model predictions are confident."
elif uncertainty_score < 0.6:
    unc_state = "🟠 MEDIUM"
    unc_msg = "Prediction confidence is decreasing."
else:
    unc_state = "🔴 HIGH"
    unc_msg = "Model is highly uncertain in its predictions."

st.markdown(f"""
### 📊 Model Health  
**{health_state}**  
{health_msg}

### 🔀 Data Drift  
**{drift_state}**  
{drift_msg}

### ❓ Model Uncertainty  
**{unc_state}**  
{unc_msg}
""")

st.divider()
# Overall recommendation
# ------------------ OVERALL SYSTEM VERDICT ------------------

if health_pct >= 75 and drift_score < 0.5:
    verdict_text = "✅ System healthy. No immediate action required."
    final_reco = "System healthy"
    bg_color = "#e6f4ea"
    border_color = "#2e7d32"

elif health_pct >= 50:
    verdict_text = "⚠️ System stable. Retraining recommended in the near future."
    final_reco = "Retraining recommended soon"
    bg_color = "#fff4e5"
    border_color = "#ef6c00"

else:
    verdict_text = "🚨 System health degraded. Immediate retraining required."
    final_reco = "Immediate retraining required"
    bg_color = "#fdecea"
    border_color = "#c62828"

st.markdown(
    f"""
    <div style="
        background-color:{bg_color};
        padding:16px;
        border-radius:8px;
        border-left:6px solid {border_color};
        font-size:18px;
        font-weight:600;
        max-width:800px;
        margin:auto;
        color:#000000;
    ">
        {verdict_text}
    </div>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# DOWNLOAD
# ------------------------------------------------------------
st.divider()

report = pd.DataFrame([{
    "model_health_pct": health_pct,
    "mae": mae,
    "drift_score": drift_score,
    "uncertainty_score": uncertainty_score,
    "health_state": health_state,
    "drift_state": drift_state,
    "uncertainty_state": unc_state,
    "final_recommendation": final_reco
}])

st.download_button(
    "⬇️📂 Download Health Report",
    report.to_csv(index=False),
    "model_health_report.csv",
    "text/csv"
)

st.caption("MODEL FAILURE FORECASTER • ML dashboard")