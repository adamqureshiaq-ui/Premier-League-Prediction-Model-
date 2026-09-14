import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pickle
import os

st.set_page_config(page_title="Football Prediction Dashboard", layout="centered")

# -----------------------------
# Safe loaders
# -----------------------------
def safe_load_pickle(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def safe_load_npy(path):
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None

# -----------------------------
# Load dataset
# -----------------------------
df_all = pd.read_csv("df_all.csv")

# -----------------------------
# Load models
# -----------------------------
print("Loading LR from:", os.path.abspath("linear_regression.pkl"))

lr_model = safe_load_pickle("linear_regression.pkl")
lr_scaler = safe_load_pickle("lr_scaler.pkl")

rf_model = safe_load_pickle("random_forest.pkl")
rf_tuned_model = safe_load_pickle("random_forest_tuned.pkl")
xgb_model = safe_load_pickle("xgb_final_model.pkl")

# -----------------------------
# Load feature lists
# -----------------------------
feature_list = safe_load_pickle("feature_list.pkl")
xgb_features_file = safe_load_pickle("xgb_final_features.pkl")

lr_features = []
rf_features = []
rf_tuned_features = []
xgb_features = []

# feature_list.pkl may contain dict of lists
if isinstance(feature_list, dict):
    lr_features = feature_list.get("Linear Regression", [])
    rf_features = feature_list.get("Random Forest", [])
    rf_tuned_features = feature_list.get("Random Forest (Tuned)", [])
elif isinstance(feature_list, list):
    lr_features = feature_list
    rf_features = feature_list
    rf_tuned_features = feature_list

# xgb features
if isinstance(xgb_features_file, list):
    xgb_features = xgb_features_file

# -----------------------------
# Load ensemble weights
# -----------------------------
ensemble_weights = safe_load_npy("ensemble_weights.npy")

# -----------------------------
# Build models dictionary
# -----------------------------
models = {
    "Linear Regression": {
        "model": lr_model,
        "scaler": lr_scaler,
        "features": lr_features
    },
    "Random Forest": {
        "model": rf_model,
        "features": rf_features
    },
    "Random Forest (Tuned)": {
        "model": rf_tuned_model,
        "features": rf_tuned_features
    },
    "XGBoost": {
        "model": xgb_model,
        "features": xgb_features
    },
    "Ensemble": {
        "weights": ensemble_weights
    }
}

# -----------------------------
# Build master feature list
# -----------------------------
feature_sets = []

for name in ["Linear Regression", "Random Forest", "Random Forest (Tuned)", "XGBoost"]:
    feats = models[name]["features"]
    if feats:
        feature_sets.append(set(feats))

if feature_sets:
    all_features = sorted(set().union(*feature_sets))
else:
    all_features = df_all.select_dtypes(include=[np.number]).columns.tolist()

# -----------------------------
# Unified feature builder
# -----------------------------
def build_features(team_home, team_away, df_all, all_features):
    home_rows = df_all[df_all["HomeTeam"] == team_home]
    away_rows = df_all[df_all["AwayTeam"] == team_away]

    home_row = home_rows.iloc[-1]
    away_row = away_rows.iloc[-1]

    combined = pd.DataFrame([home_row, away_row]).mean(numeric_only=True)
    combined = combined.to_frame().T

    for col in all_features:
        if col not in combined.columns:
            combined[col] = 0.0

    return combined[all_features]

# -----------------------------
# Prediction logic
# -----------------------------
def predict_probabilities(team_home, team_away):
    X_new = build_features(team_home, team_away, df_all, all_features)

    model_probs = []
    model_names = []

    # LR
    if models["Linear Regression"]["model"] is not None and models["Linear Regression"]["features"]:
        X_lr = X_new[models["Linear Regression"]["features"]]
        if models["Linear Regression"]["scaler"] is not None:
            X_lr = models["Linear Regression"]["scaler"].transform(X_lr)
        p_lr = models["Linear Regression"]["model"].predict_proba(X_lr)[0]
        model_probs.append(p_lr)
        model_names.append("Linear Regression")

    # RF
    if models["Random Forest"]["model"] is not None and models["Random Forest"]["features"]:
        X_rf = X_new[models["Random Forest"]["features"]]
        p_rf = models["Random Forest"]["model"].predict_proba(X_rf)[0]
        model_probs.append(p_rf)
        model_names.append("Random Forest")

    # RF Tuned
    if models["Random Forest (Tuned)"]["model"] is not None and models["Random Forest (Tuned)"]["features"]:
        X_rft = X_new[models["Random Forest (Tuned)"]["features"]]
        p_rft = models["Random Forest (Tuned)"]["model"].predict_proba(X_rft)[0]
        model_probs.append(p_rft)
        model_names.append("Random Forest (Tuned)")

    # XGB
    if models["XGBoost"]["model"] is not None and models["XGBoost"]["features"]:
        X_xgb = X_new[models["XGBoost"]["features"]]
        p_xgb = models["XGBoost"]["model"].predict_proba(X_xgb)[0]
        model_probs.append(p_xgb)
        model_names.append("XGBoost")

    # Ensemble weights
    n = len(model_probs)

    if models["Ensemble"]["weights"] is None:
        weights = np.ones(n) / n
    else:
        w = np.array(models["Ensemble"]["weights"])
        if len(w) != n:
            weights = np.ones(n) / n
        else:
            weights = w / w.sum()

    # Weighted sum
    p_final = np.zeros(3)
    for w, p in zip(weights, model_probs):
        p_final += w * p

    p_final /= p_final.sum()

    df_probs = pd.DataFrame({
        "Outcome": ["Home Win", "Draw", "Away Win"],
        "Probability": [p_final[2], p_final[1], p_final[0]]
    })

    return df_probs

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("⚽ Football Prediction Dashboard")

teams = sorted(df_all["HomeTeam"].unique())

col1, col2 = st.columns(2)
home_team = col1.selectbox("Home Team", teams)
away_team = col2.selectbox("Away Team", teams)

if st.button("Predict Match"):
    try:
        df_probs = predict_probabilities(home_team, away_team)

        df_display = df_probs.copy()
        df_display["Probability"] = (df_display["Probability"] * 100).round(2).astype(str) + " %"

        st.subheader("Predicted Probabilities")
        st.table(df_display.set_index("Outcome"))

        chart = alt.Chart(df_probs).mark_bar(size=50).encode(
            x=alt.X("Outcome:N"),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
            color="Outcome:N"
        )

        st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")
