import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os
import traceback
import json
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import time 

st.set_page_config(page_title="Football Prediction Dashboard", layout="centered")

# -----------------------------
# Safe loaders
# -----------------------------
def safe_load_joblib(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Failed to load joblib file: {path}")
            st.code(traceback.format_exc())
            return None
    return None

def safe_load_npy(path):
    if os.path.exists(path):
        try:
            return np.load(path, allow_pickle=True)
        except Exception as e:
            st.error(f"Failed to load npy file: {path}")
            st.code(traceback.format_exc())
            return None
    return None

# -----------------------------
# Data load (no caching)
# -----------------------------
def load_dataframe():
    return pd.read_csv("df_all.csv")

# -----------------------------
# Model load (no caching, fully guarded)
# -----------------------------
def load_models():
    # Random Forest models
    rf_model = safe_load_joblib("random_forest.pkl")
    rf_tuned_model = safe_load_joblib("random_forest_tuned.pkl")

    # XGBoost JSON model (Booster)
    xgb_model = None
    if os.path.exists("xgb_final_model.json"):
        try:
            xgb_model = xgb.Booster()
            xgb_model.load_model("xgb_final_model.json")
        except Exception as e:
            st.error("XGBoost JSON model failed to load:")
            st.code(traceback.format_exc())
            xgb_model = None

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

    # XGB features from JSON
    xgb_features = []
    if os.path.exists("xgb_final_features.json"):
        try:
            with open("xgb_final_features.json", "r") as f:
                xgb_features = json.load(f)
        except Exception as e:
            st.error("Failed to load xgb_final_features.json:")
            st.code(traceback.format_exc())
            xgb_features = []

    # Ensemble weights
    ensemble_weights = safe_load_npy("ensemble_weights.npy")

    models = {
        "Random Forest": {"model": rf_model, "features": rf_features},
        "Random Forest (Tuned)": {"model": rf_tuned_model, "features": rf_tuned_features},
        "XGBoost": {"model": xgb_model, "features": xgb_features},
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

# If XGBoost is missing, just warn, don't stop the whole app
if models["XGBoost"]["model"] is None:
    st.warning("XGBoost model not loaded. Predictions will use only available models.")

# -----------------------------
# Build master feature list
# -----------------------------
feature_sets = []
for name in ["Random Forest", "Random Forest (Tuned)", "XGBoost"]:
    feats = models[name]["features"]
    if feats:
        feature_sets.append(set(feats))

if feature_sets:
    all_features = sorted(set().union(*feature_sets))
else:
    all_features = df_all.select_dtypes(include=[np.number]).columns.tolist()

if not any(models[name]["model"] is not None for name in ["Random Forest", "Random Forest (Tuned)", "XGBoost"]):
    st.warning(
        "No model files were found (random_forest.pkl, random_forest_tuned.pkl, "
        "xgb_final_model.json). Predictions will not work until at least one is present "
        "in the app's working directory."
    )

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
# Prediction logic
# -----------------------------
def predict_probabilities(team_home, team_away):
    X_new = build_features(team_home, team_away, df_all, all_features)

    model_probs = []
    model_names = []

    # Sklearn models
    for name in ["Random Forest", "Random Forest (Tuned)"]:
        model_info = models[name]
        if model_info["model"] is not None and model_info["features"]:
            X_sub = X_new[model_info["features"]]
            p = model_info["model"].predict_proba(X_sub)[0]
            model_probs.append(p)
            model_names.append(name)

    # XGBoost Booster (if available)
    xgb_model = models["XGBoost"]["model"]
    xgb_features = models["XGBoost"]["features"]
    if xgb_model is not None and xgb_features:
        X_sub = X_new[xgb_features]
        dmatrix = xgb.DMatrix(X_sub)
        p = xgb_model.predict(dmatrix)[0]  # assumes 3-class output
        model_probs.append(p)
        model_names.append("XGBoost")

    n = len(model_probs)
    if n == 0:
        raise RuntimeError(
            "No trained models are available, so a prediction can't be made. "
            "Check that the model files are present."
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

    return df_probs, model_names, X_new

def generate_natural_explanation(shap_values, feature_names):
    # shap_values: shape (1, n_features)
    shap_values = shap_values.flatten()

    # Pair each feature with its SHAP impact
    feature_impacts = list(zip(feature_names, shap_values))

    # Sort by absolute impact (biggest influence first)
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

    explanation_lines = []

    for feature, impact in feature_impacts[:5]:  # top 5 features
        direction = "increases" if impact > 0 else "decreases"
        explanation_lines.append(f"- **{feature}** {direction} the chance of a Home Win.")

    explanation = (
        "### 🧠 Why the model predicted this\n"
        "The model looks at many factors. Here are the most influential ones:\n\n"
        + "\n".join(explanation_lines)
        + "\n\nOverall, the model combines these effects to estimate the outcome of the Match."
    )

    return explanation

def compute_confidence_intervals(probabilities_list):
    """
    probabilities_list: list of 3-element arrays [Home, Draw, Away]
    """
    probs = np.array(probabilities_list)  # shape (n_models, 3)

    means = probs.mean(axis=0)
    stds = probs.std(axis=0)

    # 95% CI ≈ mean ± 2 * std
    ci_low = means - 2 * stds
    ci_high = means + 2 * stds

    return means, ci_low, ci_high


def show_shap_explanations(X_new, models):
    st.subheader("🔍 SHAP Model Explanations")

    # Random Forest SHAP
    if models["Random Forest"]["model"] is not None:
        try:
            explainer_rf = shap.TreeExplainer(models["Random Forest"]["model"])
            shap_values_rf = explainer_rf.shap_values(X_new[models["Random Forest"]["features"]])

            #st.write("SHAP values RF:", np.array(shap_values_rf).shape)

            # Correct indexing: sample=0, features=:, class=2 (Home Win)
            shap_values_rf_home = shap_values_rf[0, :, 2].reshape(1, -1)

            explanation_rf = generate_natural_explanation(
                shap_values_rf_home,
                models["Random Forest"]["features"]
            )
            st.markdown(explanation_rf)


            st.write("Random Forest Feature Impact (Home Win)")
            fig, ax = plt.subplots()
            shap.summary_plot(
                shap_values_rf_home,
                X_new[models["Random Forest"]["features"]],
                show=False
            )
            st.pyplot(fig)

        except Exception:
            st.warning("SHAP failed for Random Forest.")
            st.code(traceback.format_exc())

    # XGBoost SHAP
    if models["XGBoost"]["model"] is not None:
        try:
            explainer_xgb = shap.TreeExplainer(models["XGBoost"]["model"])
            shap_values_xgb = explainer_xgb.shap_values(X_new[models["XGBoost"]["features"]])

            #st.write("XGB SHAP shape:", np.array(shap_values_xgb).shape)

            # Correct indexing: sample=0, features=:, class=2 (Home Win)
            shap_values_xgb_home = shap_values_xgb[0, :, 2].reshape(1, -1)

            explanation_xgb = generate_natural_explanation(
                shap_values_xgb_home,
                models["XGBoost"]["features"]
            )

            st.markdown(explanation_xgb)

            st.write("XGBoost Feature Impact (Home Win)")
            fig, ax = plt.subplots()
            shap.summary_plot(
                shap_values_xgb_home,
                X_new[models["XGBoost"]["features"]],
                show=False
            )
            st.pyplot(fig)

        except Exception:
            st.warning("SHAP failed for XGBoost.")
            st.code(traceback.format_exc())


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("⚽ Football Prediction Dashboard")

teams = sorted(set(df_all["HomeTeam"].unique()) | set(df_all["AwayTeam"].unique()))

col1, col2 = st.columns(2)
home_team = col1.selectbox("Home Team", teams)
away_team = col2.selectbox("Away Team", teams)

if st.button("Predict Match"):
    if home_team == away_team:
        st.error("Please choose two different teams.")
    else:
        try:
            df_probs, model_names, X_new = predict_probabilities(home_team, away_team)

            st.caption(f"Models used: {', '.join(model_names)}")

            df_display = df_probs.copy()
            df_display["Probability"] = (df_display["Probability"] * 100).round(2).astype(str) + " %"

            st.subheader("Predicted Probabilities")
            st.table(df_display.set_index("Outcome"))

            # Reorder outcomes for chart
            df_probs_chart = df_probs.copy()
            df_probs_chart["Outcome"] = pd.Categorical(
                df_probs_chart["Outcome"],
                categories=["Home Win", "Draw", "Away Win"],
                ordered=True
            )

            chart = alt.Chart(df_probs_chart).mark_bar(size=50).encode(
                x=alt.X("Outcome:N", sort=["Home Win", "Draw", "Away Win"]),
                y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
                color="Outcome:N",
            )

            # Animated probability bar chart
            placeholder = st.empty()

            steps = 30  # number of animation frames
            for i in range(steps + 1):
                df_anim = df_probs_chart.copy()
                df_anim["Probability"] = df_anim["Probability"] * (i / steps)

                chart_anim = alt.Chart(df_anim).mark_bar(size=50).encode(
                    x=alt.X("Outcome:N", sort=["Home Win", "Draw", "Away Win"]),
                    y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
                    color="Outcome:N",
                )

                placeholder.altair_chart(chart_anim, use_container_width=True)
                time.sleep(0.03)


            # -------------------------------
            # CONFIDENCE INTERVALS (RF only)
            # -------------------------------
            all_model_probs = []

            for model_name in model_names:
                model = models[model_name]["model"]
                features = models[model_name]["features"]

                # Only sklearn models support predict_proba
                if hasattr(model, "predict_proba"):
                    p = model.predict_proba(X_new[features])[0]  # [Away, Draw, Home]
                    all_model_probs.append(p)

            # Only compute CI if we have at least 2 models
            if len(all_model_probs) >= 2:
                means, ci_low, ci_high = compute_confidence_intervals(all_model_probs)

                st.subheader("📏 Confidence Intervals (Model Uncertainty)")

                ci_df = pd.DataFrame({
                    "Outcome": ["Away Win", "Draw", "Home Win"],
                    "Mean Probability": means,
                    "Lower CI": ci_low,
                    "Upper CI": ci_high
                })

                ci_df["Mean Probability"] = (ci_df["Mean Probability"] * 100).round(2).astype(str) + " %"
                ci_df["Lower CI"] = (ci_df["Lower CI"] * 100).round(2).astype(str) + " %"
                ci_df["Upper CI"] = (ci_df["Upper CI"] * 100).round(2).astype(str) + " %"

                st.table(ci_df)
            else:
                st.info("Confidence intervals require at least two models with predict_proba().")

            # SHAP explanations
            show_shap_explanations(X_new, models)

        except (ValueError, RuntimeError) as e:
            st.error(str(e))
        except Exception:
            st.error("Unexpected error during prediction:")
            st.code(traceback.format_exc())
