import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# 1. PAGE CONFIGURATION & CLEAN LIGHT THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="Bengaluru Real Estate Price Estimator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #0F172A;
        --navy-light: #1E293B;
        --teal: #0D9488;
        --teal-dark: #0F766E;
        --teal-soft: #CCFBF1;
        --text: #1E293B;
        --muted: #475569;
        --border: #CBD5E1;
        --bg-main: #EAF1F5;
        --btn-blue: #2563EB;
        --btn-blue-hover: #1D4ED8;
    }

    /* Overall App Background */
    .stApp {
        background-color: var(--bg-main) !important;
        background-image: none !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif;
    }

    /* Sidebar Background & Contrast Fixes */
    [data-testid="stSidebar"] {
        background-color: #F1F5F9 !important;
        border-right: 1px solid #CBD5E1 !important;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #0F172A !important;
        font-weight: 600 !important;
    }

    /* Headings */
    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -0.02em !important;
    }

    h1 {
        color: var(--navy) !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
        margin-bottom: 0.2rem !important;
    }

    h2, h3 {
        color: var(--navy-light) !important;
        font-weight: 700 !important;
    }

    /* Form & Input Label Contrast */
    .stApp label, 
    [data-testid="stWidgetLabel"] p,
    [data-testid="stMarkdownContainer"] p,
    .stSelectbox label, 
    .stNumberInput label {
        color: #0F172A !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }

    /* Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #CBD5E1;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 8px;
        font-weight: 700 !important;
        color: #334155 !important;
        font-size: 0.95rem !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15);
    }
    
    .stTabs [aria-selected="true"] p {
        color: #FFFFFF !important;
    }

    /* Input Field Styling */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #FFFFFF !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: #0F172A !important;
    }

    div[data-baseweb="select"] *, 
    div[data-baseweb="input"] * {
        color: #0F172A !important;
        font-weight: 600 !important;
    }

    /* Form Container */
    [data-testid="stForm"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 16px !important;
        padding: 1.75rem !important;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05) !important;
    }

    /* Primary Button */
    .st-key-predict_btn [data-testid="stBaseButton-primary"] {
        width: 100%;
        min-height: 50px;
        background-color: var(--btn-blue) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.25) !important;
        transition: all 0.2s ease !important;
    }

    .st-key-predict_btn [data-testid="stBaseButton-primary"]:hover {
        background-color: var(--btn-blue-hover) !important;
        transform: translateY(-1px);
        box-shadow: 0 8px 22px rgba(37, 99, 235, 0.35) !important;
    }

    /* Price Result Card */
    .result-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #FFFFFF;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.15);
        text-align: center;
        margin-top: 0.5rem;
    }

    .result-card h3 {
        color: var(--teal-soft) !important;
        font-size: 0.95rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }

    .result-card .price-value {
        font-family: 'Manrope', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        color: #FFFFFF;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--teal-dark) !important;
        font-family: 'Manrope', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.6rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. CACHED ARTIFACT LOADERS & HELPERS
# ==============================================================================
@st.cache_resource
def load_all_models():
    """Loads saved non-XGBoost joblib models and XGBoost JSON model safely."""
    loaded_models = {}
    
    if os.path.exists("other_models.pkl"):
        try:
            joblib_models = joblib.load("other_models.pkl")
            if isinstance(joblib_models, dict):
                loaded_models.update(joblib_models)
        except Exception as e:
            st.error(f"Error loading 'other_models.pkl': {e}")
            
    if os.path.exists("xgboost_model.json"):
        try:
            import xgboost as xgb
            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model("xgboost_model.json")
            loaded_models["XGBoost"] = xgb_model
        except Exception as e:
            st.error(f"Error loading 'xgboost_model.json': {e}")
            
    return loaded_models

@st.cache_data
def load_data_columns():
    """Loads expected training feature order."""
    if os.path.exists("columns.json"):
        with open("columns.json", "r") as f:
            return json.load(f).get("data_columns", [])
    return []

@st.cache_data
def load_metrics():
    """Loads precomputed performance metrics."""
    if os.path.exists("model_metrics.json"):
        with open("model_metrics.json", "r") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_test_artifacts():
    """Optional loader for actual test targets and predictions if available."""
    test_data = {}
    if os.path.exists("y_test.npy"):
        test_data["y_true"] = np.load("y_test.npy")
    if os.path.exists("test_predictions.json"):
        with open("test_predictions.json", "r") as f:
            test_data["predictions"] = json.load(f)
    return test_data

# Initialize artifacts
models = load_all_models()
data_columns = load_data_columns()
metrics_data = load_metrics()
test_artifacts = load_test_artifacts()

# Extract location features from columns.json
numeric_cols = ["total_sqft", "bath", "balcony", "bhk"]
locations = [c for c in data_columns if c not in numeric_cols]

# ==============================================================================
# 3. HEADER & SIDEBAR LAYOUT
# ==============================================================================
st.markdown("<h1>🏠 Bengaluru Real Estate Price Estimator</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='margin-bottom: 1.5rem; color: #475569; font-size: 1rem;'>"
    "An interactive machine learning application for predicting property valuation in Bengaluru "
    "using tabular models trained on historical real estate data."
    "</p>",
    unsafe_allow_html=True
)

# Sidebar - Model Selection & System Information
with st.sidebar:
    st.markdown("### ⚙️ Model Selection")
    
    available_model_names = list(models.keys()) if models else ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"]
    selected_algorithm = st.selectbox(
        "Active Machine Learning Model",
        options=available_model_names,
        index=0,
        help="Choose the machine learning algorithm to compute price estimates."
    )
    
    st.markdown("---")
  
    
# Main Navigation Tabs
tab_pred, tab_comp, tab_eval = st.tabs([
    "🏠 Property Price Estimator", 
    "📊 Model Performance & Comparison", 
    "📉 Residual & Error Analysis"
])

# ==============================================================================
# TAB 1: HOUSE DETAILS & PREDICTION FORM
# ==============================================================================
with tab_pred:
    col_form, col_summary = st.columns([1.5, 1], gap="large")

    with col_form:
        st.markdown("### Property Details")
        
        if not data_columns:
            st.error("⚠️ Column specifications missing (`columns.json`). Unable to render form.")
        else:
            with st.form("house_prediction_form", border=True):
                r1_col1, r1_col2 = st.columns(2)
                with r1_col1:
                    selected_location = st.selectbox(
                        "Location",
                        options=sorted(locations, key=str.lower) if locations else ["Other"],
                        help="Select the Bengaluru neighborhood or locality."
                    )
                with r1_col2:
                    sqft_val = st.number_input(
                        "Total Area (sq. ft.)",
                        min_value=300,
                        max_value=20000,
                        value=1200,
                        step=50,
                        help="Carpet or super built-up area in square feet. Example: 1200 sq. ft."
                    )

                r2_col1, r2_col2 = st.columns(2)
                with r2_col1:
                    bhk_val = st.selectbox(
                        "Bedrooms (BHK)",
                        options=[1, 2, 3, 4, 5, 6, 7, 8],
                        index=1,
                        help="Number of bedrooms, hall, and kitchen."
                    )
                with r2_col2:
                    bath_val = st.selectbox(
                        "Bathrooms",
                        options=[1, 2, 3, 4, 5, 6, 7, 8],
                        index=1,
                        help="Number of bathrooms in the property."
                    )

                balcony_val = st.selectbox(
                    "Balconies",
                    options=[0, 1, 2, 3, 4, 5],
                    index=1,
                    help="Number of attached balconies."
                )

                st.markdown("<br>", unsafe_allow_html=True)

                submit_predict = st.form_submit_button(
                    "Predict House Price",
                    type="primary",
                    key="predict_btn",
                    use_container_width=True
                )

    with col_summary:
        st.markdown("### Estimate Price")
        
        if submit_predict:
            if sqft_val < 300 or sqft_val > 20000:
                st.error("Please enter a total area between 300 and 20,000 sq. ft.")
            elif selected_algorithm not in models:
                st.error(f"Selected model '{selected_algorithm}' is not loaded. Verify your `.pkl` or `.json` artifacts.")
            else:
                input_df = pd.DataFrame(0.0, index=[0], columns=data_columns)
                
                if "total_sqft" in input_df.columns: input_df.loc[0, "total_sqft"] = float(sqft_val)
                if "bath" in input_df.columns: input_df.loc[0, "bath"] = float(bath_val)
                if "balcony" in input_df.columns: input_df.loc[0, "balcony"] = float(balcony_val)
                if "bhk" in input_df.columns: input_df.loc[0, "bhk"] = float(bhk_val)
                
                if selected_location in input_df.columns:
                    input_df.loc[0, selected_location] = 1.0

                try:
                    active_model = models[selected_algorithm]
                    log_price_pred = active_model.predict(input_df)[0]
                    final_price_lakhs = max(0.0, float(np.expm1(log_price_pred)))
                    
                    if final_price_lakhs >= 100:
                        crores_val = final_price_lakhs / 100.0
                        formatted_price = f"₹ {crores_val:.2f} Cr"
                    else:
                        formatted_price = f"₹ {final_price_lakhs:.2f} Lakhs"
                    
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <h3>Estimated Valuation</h3>
                            <div class="price-value">{formatted_price}</div>
                            <p style="color: #94A3B8; margin-top: 0.5rem; font-size: 0.85rem;">
                                Model: <strong>{selected_algorithm}</strong>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("#### Property Specifications")
                    st.info(
                        f"• **Location**: {selected_location}\n"
                        f"• **Total Area**: {sqft_val:,} sq. ft. (~₹{(final_price_lakhs*100000)/sqft_val:.0f} / sq. ft.)\n"
                        f"• **Configuration**: {bhk_val} BHK | {bath_val} Bath | {balcony_val} Balcony"
                    )

                except Exception as err:
                    st.error(f"Inference Exception: {err}")
        else:
            st.info("👈 Enter property details and click **Predict House Price** to compute valuation.")

# ==============================================================================
# TAB 2: MODEL EVALUATION & COMPARISON DASHBOARD
# ==============================================================================
with tab_comp:
    st.markdown("### Selected Model Metrics")
    
    if metrics_data and selected_algorithm in metrics_data:
        alg_metrics = metrics_data[selected_algorithm]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean Absolute Error (MAE)", f"{alg_metrics.get('MAE', 0):.3f} Lakhs", help="Average absolute difference between actual and predicted prices.")
        m2.metric("Root Mean Squared Error (RMSE)", f"{alg_metrics.get('RMSE', 0):.3f} Lakhs", help="Square root of average squared errors; gives heavier weight to large deviations.")
        m3.metric("R² Score", f"{alg_metrics.get('R2 Score', 0):.4f}", help="Proportion of variance in target variable explained by the model (Max 1.0).")
    else:
        st.warning("Precomputed metrics missing for the selected algorithm in `model_metrics.json`.")

    st.markdown("---")
    st.markdown("### Cross-Algorithm Benchmark")
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data).T.reset_index()
        metrics_df.rename(columns={"index": "Algorithm"}, inplace=True)
        
        metrics_df["Color"] = metrics_df["Algorithm"].apply(
            lambda x: "#0D9488" if x == selected_algorithm else "#94A3B8"
        )

        c_mae, c_rmse, c_r2 = st.columns(3)

        with c_mae:
            fig_mae = px.bar(
                metrics_df, x="Algorithm", y="MAE",
                title="MAE (Lower is Better)",
                color="Algorithm",
                color_discrete_map={row["Algorithm"]: row["Color"] for _, row in metrics_df.iterrows()},
                text_auto=".3f"
            )
            fig_mae.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_mae, use_container_width=True)

        with c_rmse:
            fig_rmse = px.bar(
                metrics_df, x="Algorithm", y="RMSE",
                title="RMSE (Lower is Better)",
                color="Algorithm",
                color_discrete_map={row["Algorithm"]: row["Color"] for _, row in metrics_df.iterrows()},
                text_auto=".3f"
            )
            fig_rmse.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_rmse, use_container_width=True)

        with c_r2:
            fig_r2 = px.bar(
                metrics_df, x="Algorithm", y="R2 Score",
                title="R² Score (Higher is Better)",
                color="Algorithm",
                color_discrete_map={row["Algorithm"]: row["Color"] for _, row in metrics_df.iterrows()},
                text_auto=".4f"
            )
            fig_r2.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_r2, use_container_width=True)

# ==============================================================================
# TAB 3: RESIDUAL & ERROR ANALYSIS
# ==============================================================================
with tab_eval:
    st.markdown("### Actual vs. Predicted & Residual Distribution")
    
    y_true = test_artifacts.get("y_true")
    predictions_dict = test_artifacts.get("predictions", {})

    if y_true is not None and selected_algorithm in predictions_dict:
        y_pred = np.array(predictions_dict[selected_algorithm])
        residuals = y_true - y_pred

        col_scat, col_hist = st.columns(2)

        with col_scat:
            st.markdown("#### Actual vs. Predicted Prices")
            fig_scatter = go.Figure()

            fig_scatter.add_trace(go.Scatter(
                x=y_true, y=y_pred,
                mode='markers',
                marker=dict(color='#0D9488', opacity=0.6, size=6),
                name='Properties'
            ))

            max_val = max(np.max(y_true), np.max(y_pred))
            fig_scatter.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode='lines',
                line=dict(color='#EF4444', dash='dash', width=2),
                name='Ideal Fit (Y = X)'
            ))

            fig_scatter.update_layout(
                xaxis_title="Actual Price (₹ Lakhs)",
                yaxis_title="Predicted Price (₹ Lakhs)",
                height=400,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_hist:
            st.markdown("#### Prediction Error Distribution (Residuals)")
            fig_hist = px.histogram(
                x=residuals,
                nbins=40,
                labels={'x': 'Residual (Actual - Predicted in ₹ Lakhs)'},
                title="Error Distribution",
                color_discrete_sequence=['#2563EB']
            )
            fig_hist.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)

    else:
        st.warning("⚠️ **Test Set Artifacts Required for Plotting**")
        st.info(
            "To view **Actual vs. Predicted Scatter Plots** and **Residual Histograms**, save your test evaluation targets during model training:\n\n"
            "1. **`y_test.npy`**: NumPy array containing ground-truth price targets (in ₹ Lakhs).\n"
            "2. **`test_predictions.json`**: A JSON dictionary mapping algorithm names to prediction arrays on the test set."
        )