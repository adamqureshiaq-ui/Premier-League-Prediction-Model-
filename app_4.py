import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os
import traceback

st.set_page_config(page_title="Football Prediction Dashboard (RF only)", layout="centered")

# -----------------------------
# Safe loaders
# -----------------------------
def safe_load_joblib(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            st.error(f"Failed to load joblib file: {path}")
            st.code(traceback.format_exc())
            return None
    return None

def safe_load_npy(path):
    if os.path.exists(path):
        try:
            return np.load(path, allow_pickle=True)
        except Exception:
            st.error(f"Failed to load npy file: {path}")
            st.code(traceback.format_exc())
            return None
    return None

# -----------------------------
# Data load
# -----------------------------
def load_dataframe():
    return pd.read_csv("df_all.csv")

# -----------------------------
# Model load (RF only)
# -----------------------------
def load_models():
    # Random Forest models
    rf_model = safe_load_joblib("random_forest.pkl")
    rf_tuned_model = safe_load_joblib("random_forest_tuned.pkl")

    # Feature list for RF models
    feature_list = safe_load_joblib("feature_list.pkl")

    rf_features = []
    rf_tuned_features = []

    if isinstance(feature_list, dict):
        rf_features = feature_list.get("Random Forest", [])
        rf_tuned_features = feature_list.get("Random Forest (Tuned)", [])
    elif isinstance(feature_list, list):
        rf_features = feature_list
        rf_tuned_features = feature_list

    # Ensemble weights (optional)
    ensemble_weights = safe_load_npy("ensemble_weights.npy")

    models = {
        "Random Forest": {"model": rf_model, "features": rf_features},
        "Random Forest (Tuned)": {"model": rf_tuned_model, "features": rf_tuned_features},
        "Ensemble": {"weights": ensemble_weights},
    }
    return models

# -----------------------------
# Load data and models
# -----------------------------
if not os.path.exists("df_all.csv"):
    st.error("df_all.csv not found. Please make sure it is in the app's working directory.")
    st.stop()

try:
    df_all = load_dataframe()
except Exception:
    st.error("Data loading crashed:")
    st.code(traceback.format_exc())
    st.stop()

try:
    models = load_models()
except Exception:
    st.error("Model loading crashed:")
    st.code(traceback.format_exc())
    st.stop()

if not any(models[name]["model"] is not None for name in ["Random Forest", "Random Forest (Tuned)"]):
    st.error(
        "No RF models were loaded. Make sure random_forest.pkl and/or "
        "random_forest_tuned.pkl are in the working directory."
    )
    st.stop()

# -----------------------------
# Build master feature list
# -----------------------------
feature_sets = []
for name in ["Random Forest", "Random Forest (Tuned)"]:
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

    if home_rows.empty:
        raise ValueError(f"No historical rows found where '{team_home}' played at home.")
    if away_rows.empty:
        raise ValueError(f"No historical rows found where '{team_away}' played away.")

    home_row = home_rows.iloc[-1]
    away_row = away_rows.iloc[-1]

    combined = pd.DataFrame([home_row, away_row]).mean(numeric_only=True)
    combined = combined.to_frame().T

    for col in all_features:
        if col not in combined.columns:
            combined[col] = 0.0

    return combined[all_features]

# -----------------------------
# Prediction logic (RF only)
# -----------------------------
def predict_probabilities(team_home, team_away):
    X_new = build_features(team_home, team_away, df_all, all_features)

    model_probs = []
    model_names = []

    for name in ["Random Forest", "Random Forest (Tuned)"]:
        model_info = models[name]
        if model_info["model"] is not None and model_info["features"]:
            X_sub = X_new[model_info["features"]]
            p = model_info["model"].predict_proba(X_sub)[0]
            model_probs.append(p)
            model_names.append(name)

    n = len(model_probs)
    if n == 0:
        raise RuntimeError(
            "No trained RF models are available, so a prediction can't be made. "
            "Check that the RF .pkl files are present."
        )

    ensemble_weights = models["Ensemble"]["weights"]
    if ensemble_weights is None:
        weights = np.ones(n) / n
    else:
        w = np.array(ensemble_weights, dtype=float)
        if len(w) != n:
            weights = np.ones(n) / n
        else:
            weights = w / w.sum()

    p_final = np.zeros(3)
    for w, p in zip(weights, model_probs):
        p_final += w * p
    p_final /= p_final.sum()

    df_probs = pd.DataFrame({
        "Outcome": ["Home Win", "Draw", "Away Win"],
        "Probability": [p_final[2], p_final[1], p_final[0]],
    })

    return df_probs, model_names

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("⚽ Football Prediction Dashboard (RF only)")

teams = sorted(set(df_all["HomeTeam"].unique()) | set(df_all["AwayTeam"].unique()))

col1, col2 = st.columns(2)
home_team = col1.selectbox("Home Team", teams)
away_team = col2.selectbox("Away Team", teams)

if st.button("Predict Match"):
    if home_team == away_team:
        st.error("Please choose two different teams.")
    else:
        try:
            df_probs, model_names = predict_probabilities(home_team, away_team)

            st.caption(f"Models used: {', '.join(model_names)}")

            df_display = df_probs.copy()
            df_display["Probability"] = (df_display["Probability"] * 100).round(2).astype(str) + " %"

            st.subheader("Predicted Probabilities")
            st.table(df_display.set_index("Outcome"))

            chart = alt.Chart(df_probs).mark_bar(size=50).encode(
                x=alt.X("Outcome:N"),
                y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
                color="Outcome:N",
            )

            st.altair_chart(chart, use_container_width=True)

        except (ValueError, RuntimeError) as e:
            st.error(str(e))
        except Exception:
            st.error("Unexpected error during prediction:")
            st.code(traceback.format_exc())
