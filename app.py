"""
app.py — Interactive California House Price Prediction Dashboard & Inference App
Built with Streamlit, Scikit-learn, XGBoost, and Altair / PyDeck.
"""
import os
import io
import time
import numpy as np
import pandas as pd
import joblib
import yaml
import streamlit as st
import pydeck as pdk

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 15px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #10b981;
        letter-spacing: -0.5px;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
    }
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-blue {
        background: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Resource Caching & Data Helpers
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load scaler, all trained models, and configuration."""
    config_path = "config.yaml"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    models_dir = config.get("paths", {}).get("models_dir", "models/")
    
    # Load Scaler
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    # Load Models
    model_names = ["xgboost", "random_forest", "ridge", "linear_regression", "lasso", "baseline"]
    models = {}
    for name in model_names:
        p = os.path.join(models_dir, f"{name}.joblib")
        if os.path.exists(p):
            models[name] = joblib.load(p)

    return config, scaler, models


@st.cache_data
def load_raw_dataset():
    """Load raw housing CSV if available."""
    raw_path = "data/raw/housing.csv"
    if os.path.exists(raw_path):
        return pd.read_csv(raw_path)
    return None


@st.cache_data
def load_final_report_metrics():
    """Load evaluation test metrics from reports/final_report.md or default data."""
    report_path = "reports/final_report.md"
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Extract table
            if "## Test Set Results" in content:
                table_part = content.split("## Test Set Results")[1].split("## Notes")[0].strip()
                lines = [l.strip() for l in table_part.split("\n") if l.strip().startswith("|")]
                if len(lines) >= 3:
                    cols = [c.strip() for c in lines[0].split("|")[1:-1]]
                    data = []
                    for row in lines[2:]:
                        vals = [c.strip() for c in row.split("|")[1:-1]]
                        if len(vals) == len(cols):
                            data.append(vals)
                    df = pd.DataFrame(data, columns=cols)
                    for col in ["RMSE", "MAE", "R2", "CV RMSE"]:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                    return df
        except Exception:
            pass

    # Fallback to defaults from training
    return pd.DataFrame([
        {"Model": "xgboost", "RMSE": 0.1424, "MAE": 0.0998, "R2": 0.8391, "CV RMSE": 0.1425},
        {"Model": "random_forest", "RMSE": 0.1611, "MAE": 0.1138, "R2": 0.7941, "CV RMSE": 0.1602},
        {"Model": "ridge", "RMSE": 0.2142, "MAE": 0.1603, "R2": 0.6361, "CV RMSE": 0.2086},
        {"Model": "linear_regression", "RMSE": 0.2142, "MAE": 0.1603, "R2": 0.6361, "CV RMSE": 0.2086},
        {"Model": "lasso", "RMSE": 0.2816, "MAE": 0.2254, "R2": 0.3709, "CV RMSE": 0.2802},
        {"Model": "baseline", "RMSE": 0.3558, "MAE": 0.2879, "R2": -0.0042, "CV RMSE": 0.3576},
    ])


# Feature list in exact training order
FEATURE_ORDER = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", 
    "Population", "AveOccup", "Latitude", "Longitude", 
    "RoomsPerHousehold", "PopPerHousehold"
]

# Preset Locations in California
CITY_PRESETS = {
    "San Francisco (Bay Area)": {"lat": 37.7749, "lon": -122.4194, "med_inc": 7.5, "age": 35.0, "rooms": 5.5, "bedrms": 1.1, "pop": 1200.0, "occup": 2.6},
    "Silicon Valley (San Jose)": {"lat": 37.3382, "lon": -121.8863, "med_inc": 8.5, "age": 22.0, "rooms": 6.2, "bedrms": 1.1, "pop": 1600.0, "occup": 3.1},
    "Los Angeles (Westside)": {"lat": 34.0522, "lon": -118.2437, "med_inc": 6.2, "age": 30.0, "rooms": 5.2, "bedrms": 1.1, "pop": 1500.0, "occup": 3.0},
    "San Diego (Coastal)": {"lat": 32.7157, "lon": -117.1611, "med_inc": 5.8, "age": 25.0, "rooms": 5.6, "bedrms": 1.1, "pop": 1400.0, "occup": 2.8},
    "Sacramento (Central Valley)": {"lat": 38.5816, "lon": -121.4944, "med_inc": 3.8, "age": 28.0, "rooms": 5.4, "bedrms": 1.0, "pop": 1350.0, "occup": 2.9},
    "Fresno (Central Valley)": {"lat": 36.7468, "lon": -119.7726, "med_inc": 2.8, "age": 24.0, "rooms": 5.0, "bedrms": 1.0, "pop": 1700.0, "occup": 3.3},
    "Lake Tahoe (Sierra Nevada)": {"lat": 39.0968, "lon": -120.0324, "med_inc": 5.1, "age": 20.0, "rooms": 6.0, "bedrms": 1.2, "pop": 800.0, "occup": 2.4},
    "Custom Coordinates": None,
}


def predict_price(input_features: dict, model, scaler):
    """
    Compute engineered features, scale, predict log target,
    and convert to USD price.
    """
    med_inc = float(input_features["MedInc"])
    house_age = float(input_features["HouseAge"])
    ave_rooms = float(input_features["AveRooms"])
    ave_bedrms = float(input_features["AveBedrms"])
    population = float(input_features["Population"])
    ave_occup = float(input_features["AveOccup"])
    latitude = float(input_features["Latitude"])
    longitude = float(input_features["Longitude"])

    # Feature Engineering (mirroring engineer.py)
    rooms_per_household = ave_rooms / (house_age if house_age > 0 else 1.0)
    pop_per_household = population / (ave_occup if ave_occup > 0 else 1.0)

    row = np.array([[
        med_inc, house_age, ave_rooms, ave_bedrms,
        population, ave_occup, latitude, longitude,
        rooms_per_household, pop_per_household
    ]])

    scaled_row = scaler.transform(row)
    log_pred = float(model.predict(scaled_row)[0])
    
    # Target was log1p(MedHouseVal in $100k)
    med_house_val = float(np.expm1(log_pred))
    # Cap negative or extreme artifacts
    med_house_val = max(0.15, med_house_val)
    price_usd = med_house_val * 100000.0

    return price_usd, med_house_val, log_pred, {
        "RoomsPerHousehold": rooms_per_household,
        "PopPerHousehold": pop_per_household
    }


# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
config, scaler, models = load_artifacts()

st.title("🏡 California House Price Prediction Platform")
st.markdown(
    "Predict median district home values in California using production-grade machine learning models trained via the **D.A.T.A.** protocol."
)

if not models or scaler is None:
    st.error("⚠️ Trained models or scaler not found in `models/`. Run `python main.py` first to generate pipeline artifacts.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Interactive Estimator",
    "📊 Model Performance & Leaderboard",
    "🗺️ Exploratory Data Analysis",
    "📁 Batch Prediction"
])

# =============================================================================
# TAB 1: INTERACTIVE ESTIMATOR
# =============================================================================
with tab1:
    col_input, col_result = st.columns([1.1, 1], gap="large")

    with col_input:
        st.subheader("1. Property & Demographic Parameters")

        # Preset selector
        preset_choice = st.selectbox(
            "📍 Quick City Preset",
            options=list(CITY_PRESETS.keys()),
            index=0,
            help="Select a California region to autofill typical demographics and coordinates."
        )

        preset_vals = CITY_PRESETS.get(preset_choice)

        with st.expander("📍 Geographic Location", expanded=True):
            geo_col1, geo_col2 = st.columns(2)
            with geo_col1:
                lat = st.number_input(
                    "Latitude",
                    min_value=32.0, max_value=42.5,
                    value=preset_vals["lat"] if preset_vals else 37.7749,
                    step=0.01,
                    format="%.4f"
                )
            with geo_col2:
                lon = st.number_input(
                    "Longitude",
                    min_value=-124.5, max_value=-114.0,
                    value=preset_vals["lon"] if preset_vals else -122.4194,
                    step=0.01,
                    format="%.4f"
                )

        with st.expander("💼 Economic & Property Attributes", expanded=True):
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                med_inc = st.slider(
                    "Median Block Income ($10k USD)",
                    min_value=0.5, max_value=15.0,
                    value=preset_vals["med_inc"] if preset_vals else 5.0,
                    step=0.1,
                    help="Median income in tens of thousands (e.g. 5.0 = $50,000/year)"
                )
                ave_rooms = st.slider(
                    "Average Rooms per Home",
                    min_value=1.0, max_value=12.0,
                    value=preset_vals["rooms"] if preset_vals else 5.5,
                    step=0.1
                )
                ave_bedrms = st.slider(
                    "Average Bedrooms per Home",
                    min_value=0.5, max_value=5.0,
                    value=preset_vals["bedrms"] if preset_vals else 1.1,
                    step=0.05
                )
            with p_col2:
                house_age = st.slider(
                    "Median House Age (Years)",
                    min_value=1, max_value=52,
                    value=int(preset_vals["age"]) if preset_vals else 28,
                    step=1
                )
                population = st.number_input(
                    "Block Population",
                    min_value=10, max_value=25000,
                    value=int(preset_vals["pop"]) if preset_vals else 1400,
                    step=50
                )
                ave_occup = st.slider(
                    "Average Household Occupancy",
                    min_value=1.0, max_value=8.0,
                    value=preset_vals["occup"] if preset_vals else 3.0,
                    step=0.1
                )

        st.subheader("2. Model Selection")
        selected_model_name = st.selectbox(
            "Choose Estimator Model",
            options=list(models.keys()),
            index=0 if "xgboost" in models else 0,
            format_func=lambda x: f"{x.upper()} {'🌟 (Recommended Best)' if x == 'xgboost' else ''}"
        )

    with col_result:
        st.subheader("Prediction Results")

        input_data = {
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": lat,
            "Longitude": lon,
        }

        active_model = models[selected_model_name]
        pred_usd, med_val, log_target, eng_feats = predict_price(input_data, active_model, scaler)

        # Baseline comparison
        ca_median = 206850.0  # California dataset median price
        diff_pct = ((pred_usd - ca_median) / ca_median) * 100

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Estimated Median Home Value ({selected_model_name.upper()})</div>
            <div class="metric-value">${pred_usd:,.0f}</div>
            <div>
                <span class="badge-pill {'badge-blue' if diff_pct < 0 else ''}">
                    {'+' if diff_pct > 0 else ''}{diff_pct:.1f}% vs CA State Median ($206,850)
                </span>
                &nbsp;
                <span class="badge-pill">Model: {selected_model_name.upper()}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Multi-model Comparison Card
        with st.expander("🔍 Model Consensus Comparison", expanded=True):
            consensus_data = []
            for m_name, m_obj in models.items():
                p_usd, _, _, _ = predict_price(input_data, m_obj, scaler)
                consensus_data.append({
                    "Model": m_name.replace("_", " ").title(),
                    "Prediction ($)": p_usd,
                    "Formatted": f"${p_usd:,.0f}"
                })
            df_consensus = pd.DataFrame(consensus_data)
            
            st.bar_chart(df_consensus.set_index("Model")["Prediction ($)"])
            st.dataframe(
                df_consensus[["Model", "Formatted"]].rename(columns={"Formatted": "Estimated Price"}),
                width="stretch",
                hide_index=True
            )

        # Map display
        st.subheader("📍 Target Location Visualizer")
        map_df = pd.DataFrame([{"lat": lat, "lon": lon}])

        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=lat,
                longitude=lon,
                zoom=8,
                pitch=40,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df,
                    get_position="[lon, lat]",
                    get_color="[239, 68, 68, 200]",
                    get_radius=8000,
                    pickable=True,
                ),
            ],
            tooltip={"html": "<b>Selected Location</b><br/>Lat: {lat}<br/>Lon: {lon}"}
        ))

        # Engineered features preview
        st.caption(f"🔧 **Auto-Engineered Features:** Rooms/Age: `{eng_feats['RoomsPerHousehold']:.3f}` | Population/Occupancy: `{eng_feats['PopPerHousehold']:.1f}`")


# =============================================================================
# TAB 2: MODEL PERFORMANCE & LEADERBOARD
# =============================================================================
with tab2:
    st.subheader("🏆 Model Leaderboard & Cross-Validation Metrics")
    df_metrics = load_final_report_metrics()

    if not df_metrics.empty:
        col_m1, col_m2, col_m3 = st.columns(3)
        best_row = df_metrics.sort_values("RMSE").iloc[0]
        with col_m1:
            st.metric("Top Model", best_row["Model"].upper())
        with col_m2:
            st.metric("Best Test RMSE", f"{best_row['RMSE']:.4f}")
        with col_m3:
            st.metric("Best Test R²", f"{best_row['R2']:.4f}")

        st.dataframe(
            df_metrics.style.format({
                "RMSE": "{:.4f}",
                "MAE": "{:.4f}",
                "R2": "{:.4f}",
                "CV RMSE": "{:.4f}",
            }).highlight_min(subset=["RMSE", "MAE", "CV RMSE"], color="#064e3b")
              .highlight_max(subset=["R2"], color="#064e3b"),
            width="stretch"
        )

        st.divider()

        st.subheader("📈 Comparative Evaluation Plots")
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            comp_plot_path = "reports/eda/model_comparison.png"
            if os.path.exists(comp_plot_path):
                st.image(comp_plot_path, caption="Test RMSE Comparison across Models", width="stretch")
            else:
                st.info("Run `python main.py` to generate model comparison plots.")

        with pcol2:
            xgb_imp_path = "reports/eda/xgboost_importance.png"
            if os.path.exists(xgb_imp_path):
                st.image(xgb_imp_path, caption="XGBoost Feature Importances", width="stretch")
            else:
                rf_imp_path = "reports/eda/random_forest_importance.png"
                if os.path.exists(rf_imp_path):
                    st.image(rf_imp_path, caption="Random Forest Feature Importances", width="stretch")


# =============================================================================
# TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
with tab3:
    st.subheader("📊 Dataset Exploration & Feature Analysis")
    df_raw = load_raw_dataset()

    if df_raw is not None:
        st.write(f"Dataset Shape: `{df_raw.shape[0]:,}` rows × `{df_raw.shape[1]}` columns")
        
        with st.expander("🔍 Preview Dataset Sample (Top 10 Rows)", expanded=False):
            st.dataframe(df_raw.head(10), width="stretch")

        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            dist_path = "reports/eda/target_distribution.png"
            if os.path.exists(dist_path):
                st.image(dist_path, caption="Target Distribution (Raw vs. Log-Transformed)", width="stretch")
        with col_eda2:
            heat_path = "reports/eda/correlation_heatmap.png"
            if os.path.exists(heat_path):
                st.image(heat_path, caption="Feature Correlation Heatmap", width="stretch")

        st.subheader("🗺️ Geographic Price Distribution across California")
        # Sample for fast interactive pydeck rendering
        sample_df = df_raw.sample(min(3000, len(df_raw)), random_state=42).copy()
        sample_df["normalized_price"] = (sample_df["MedHouseVal"] - sample_df["MedHouseVal"].min()) / (
            sample_df["MedHouseVal"].max() - sample_df["MedHouseVal"].min()
        )
        # RGB based on normalized price
        sample_df["r"] = (sample_df["normalized_price"] * 255).astype(int)
        sample_df["g"] = ((1 - sample_df["normalized_price"]) * 100).astype(int)
        sample_df["b"] = ((1 - sample_df["normalized_price"]) * 220).astype(int)

        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=36.7783,
                longitude=-119.4179,
                zoom=5.5,
                pitch=30,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=sample_df,
                    get_position="[Longitude, Latitude]",
                    get_color="[r, g, b, 140]",
                    get_radius=3500,
                    pickable=True,
                ),
            ],
            tooltip={"html": "<b>MedHouseVal:</b> ${MedHouseVal}00,000<br/><b>MedInc:</b> ${MedInc}0k<br/><b>Lat/Lon:</b> {Latitude}, {Longitude}"}
        ))
    else:
        st.warning("Raw data file `data/raw/housing.csv` not found. Run `python main.py` to fetch dataset.")


# =============================================================================
# TAB 4: BATCH PREDICTION
# =============================================================================
with tab4:
    st.subheader("📁 Upload CSV for Batch Inference")
    st.markdown("Upload a CSV file containing property attributes to generate predictions for multiple locations at once.")

    # Download sample template
    sample_template = pd.DataFrame([
        {"MedInc": 8.32, "HouseAge": 41.0, "AveRooms": 6.98, "AveBedrms": 1.02, "Population": 322.0, "AveOccup": 2.55, "Latitude": 37.88, "Longitude": -122.23},
        {"MedInc": 3.87, "HouseAge": 28.0, "AveRooms": 5.40, "AveBedrms": 1.05, "Population": 1400.0, "AveOccup": 3.00, "Latitude": 34.05, "Longitude": -118.24},
        {"MedInc": 2.50, "HouseAge": 15.0, "AveRooms": 4.80, "AveBedrms": 1.10, "Population": 2200.0, "AveOccup": 3.40, "Latitude": 36.75, "Longitude": -119.77},
    ])
    csv_buffer = io.StringIO()
    sample_template.to_csv(csv_buffer, index=False)

    st.download_button(
        label="📥 Download Sample Batch Template (CSV)",
        data=csv_buffer.getvalue(),
        file_name="sample_housing_batch.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            required_cols = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
            missing_cols = [c for c in required_cols if c not in batch_df.columns]

            if missing_cols:
                st.error(f"Uploaded CSV is missing required columns: `{missing_cols}`")
            else:
                batch_model_name = st.selectbox("Model for Batch Inference", options=list(models.keys()), index=0)
                selected_batch_model = models[batch_model_name]

                if st.button("🚀 Run Batch Predictions"):
                    with st.spinner("Processing predictions..."):
                        preds_usd = []
                        for _, row in batch_df.iterrows():
                            p_usd, _, _, _ = predict_price(row.to_dict(), selected_batch_model, scaler)
                            preds_usd.append(round(p_usd, 2))

                        results_df = batch_df.copy()
                        results_df["Predicted_Price_USD"] = preds_usd
                        results_df["Formatted_Price"] = results_df["Predicted_Price_USD"].apply(lambda x: f"${x:,.2f}")

                        st.success(f"Successfully generated predictions for {len(results_df)} properties!")
                        st.dataframe(results_df, width="stretch")

                        # Download output
                        out_csv = io.StringIO()
                        results_df.to_csv(out_csv, index=False)
                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=out_csv.getvalue(),
                            file_name="predicted_housing_results.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("California Housing Price Prediction Pipeline • Powered by Scikit-Learn & XGBoost • Built with Streamlit")
