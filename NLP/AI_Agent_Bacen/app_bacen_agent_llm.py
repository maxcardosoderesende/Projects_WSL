# AI BACEN Agent SGS Time Series Web App
import textwrap
import warnings
from datetime import datetime
import os

import pandas as pd
import streamlit as st
import plotly.express as px 
from dotenv import load_dotenv
from openai import OpenAI

from src.bacen_agent_utils import (
    bacen_agent_load_similarity_openai,
    bacen_agent_plot_full,
    bacen_agent_plot_series,
    plot_acf_pacf,
    time_series_diagnostics,
    baseline_forecast
)

warnings.filterwarnings("ignore")

# -------------------------------------------------
# Environment & OpenAI initialization
# -------------------------------------------------
load_dotenv()

@st.cache_resource(show_spinner="🔌 Connecting to OpenAI...")
def load_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client = load_openai_client()

# -------------------------------------------------
# Session State Initialization
# -------------------------------------------------
for key in ["df", "code", "tab_raw", "tab_diff", "ai_text_raw", "ai_text_diff", "ai_text_lags"]:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------------------------------------
# OpenAI prompt generator
# -------------------------------------------------
def generate_prompt_from_data(prompt_text: str) -> str:
    """
    Generates AI insights from a text prompt using OpenAI GPT-4o-mini.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You are an economist specialized in BACEN (Brazilian Central Bank) time-series analysis."},
            {"role": "user", "content": prompt_text},
        ],
        max_tokens=400,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
st.markdown(
    """
    <style>
        .dashed-line {
            border-top: 2px dashed #1E90FF;
            margin-top: 6px;
            margin-bottom: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="BACEN AI Time-Series Agent", layout="wide")
st.title("Time-Series Analysis Assistant")

st.markdown(
    """
    This app lets you query **BACEN (SGS)** time-series data using natural-language prompts.
    Example queries:
    - *“Expectativa de IPCA Focus média”*
    - *“Saldo poupança SBPE”*
    - *“Taxa Selic anualizada”*
    """
)

# -------------------------------------------------
# Prompt + Dates
# -------------------------------------------------
prompt = st.text_input("🗣️ Enter your prompt:", placeholder="e.g., saldo poupanca SBPE")

col1, col2 = st.columns(2)
start_date = col1.date_input("Start date", datetime(2018, 1, 1))
end_date = col2.date_input("End date", datetime.today())

# -------------------------------------------------
# RUN ANALYSIS
# -------------------------------------------------
if st.button("🔍 Run Analysis", width="stretch"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Querying BACEN API..."):
            # ✅ Embedder removed — bacen_agent_load_similarity_openai must handle plain text
            df, code, best_key, sim = bacen_agent_load_similarity_openai(
                prompt, start=start_date, client=client
        )


        if df is None or df.empty:
            st.error("⚠️ No results found for that prompt.")
        else:
            st.success(f"Retrieved {len(df)} rows for code {code}")
            st.session_state.df = df
            st.session_state.code = code
            st.session_state.tab_raw = time_series_diagnostics(df, lags=12, use_returns=False)
            st.session_state.ai_text_raw = None
            st.session_state.ai_text_diff = None

# -------------------------------------------------
# DISPLAY SECTION (persists after rerun)
# -------------------------------------------------
if st.session_state.df is not None:
    df = st.session_state.df
    code = st.session_state.code
    tab_raw = st.session_state.tab_raw

    # ---------------- RAW SERIES ----------------
    st.markdown('<div class="dashed-line"></div>', unsafe_allow_html=True)
    st.header("Time-series preestimation diagnostics")
    st.markdown('<div class="dashed-line"></div>', unsafe_allow_html=True)

    st.subheader("Summary Statistics (Raw Series)")


    col1, col2 = st.columns(2)
    with col1:
        fig1 = bacen_agent_plot_series(df, code, use_returns=False)
        st.pyplot(fig1, width="stretch")

    with col2:
        selected_rows_raw = tab_raw[
            tab_raw["Statistic"].isin(
                ["Mean", "Standard deviation", "Skewness", "Kurtosis", "Jarque-Bera (p)", "ADF (p)"]
            )
        ]
        st.dataframe(selected_rows_raw, width="stretch", hide_index=True)

    # -------- AI INTERPRETATION (RAW SERIES) --------
    stats_dict_raw = {row["Statistic"]: row["Value"] for _, row in selected_rows_raw.iterrows()}
    prompt_llm_raw = f"""
    You are analyzing a macroeconomic time series and want to run pre-tests before modeling.

    Based on the chart (raw series) and the following summary statistics:
    {stats_dict_raw}

    Explain:
    1. What the mean and standard deviation indicate about level and volatility.
    2. How skewness and kurtosis describe the distribution shape.
    3. Explain the influence of a negative kurtosis value.
    4. Whether the series appears stationary based on the ADF result.
    5. Explain Differencing to deal with non-stationarity and why it matters.
    """

    if st.button("🧠 Generate Interpretation (Raw)", width="stretch", key="ai_raw_btn"):
        with st.spinner("Generating AI interpretation..."):
            try:
                ai_text = generate_prompt_from_data(prompt_llm_raw)
                st.session_state.ai_text_raw = ai_text
            except Exception as e:
                st.error(f"⚠️ Error generating AI interpretation: {e}")

    if st.session_state.ai_text_raw:
        st.info(st.session_state.ai_text_raw)

    # ---------------- DIFF SERIES ----------------
    st.markdown('<div class="dashed-line"></div>', unsafe_allow_html=True)
    st.subheader("Summary Statistics (Diff Series)")

    col1, col2 = st.columns(2)
    with col1:
        fig_diff = bacen_agent_plot_series(df, code, use_returns=True)
        st.pyplot(fig_diff, width="stretch")

    with col2:
        tab_diff = time_series_diagnostics(df, lags=12, use_returns=True)
        selected_rows_diff = tab_diff[
            tab_diff["Statistic"].isin(
                [
                    "Mean",
                    "Standard deviation",
                    "Skewness",
                    "Kurtosis",
                    "Jarque-Bera (p)",
                    "ADF (p)",
                    "Ljung-Box Q-test (p)",
                    "Ljung-Box Q²-test (p)",
                    "ARCH test (p)",
                ]
            )
        ]
        st.dataframe(selected_rows_diff, width="stretch", hide_index=True)

    # -------- AI INTERPRETATION (DIFF SERIES) --------
    stats_dict_diff = {row["Statistic"]: row["Value"] for _, row in selected_rows_diff.iterrows()}

    # Extract ADF p-value (handle missing key gracefully)
    adf_p_value = stats_dict_diff.get("ADF (p)", None)

    prompt_llm_diff = f"""
    After differencing, the time series becomes more stable.

    Based on these summary statistics:
    {stats_dict_diff}

    Explain:
    1. What the mean value indicates in terms of time-varying returns.
    2. Whether the series appears stationary based on the ADF p-value.
    3. Explain the Ljung-Box Q-test result and what it says about residual autocorrelation.
    """

    if st.button("🧠 Generate Interpretation (Diff)", width="stretch", key="ai_diff_btn"):
        with st.spinner("Generating AI interpretation..."):
            try:
                ai_text = generate_prompt_from_data(prompt_llm_diff)
                st.session_state.ai_text_diff = ai_text
            except Exception as e:
                st.error(f"⚠️ Error generating AI interpretation: {e}")

    if st.session_state.ai_text_diff:
        st.info(st.session_state.ai_text_diff)

    # ---------------- ACF & PACF ----------------
    st.markdown('<div class="dashed-line"></div>', unsafe_allow_html=True)
    st.subheader("Lag Definitions — ACF and PACF plots")

    fig3, metrics_raw = plot_acf_pacf(df, lags=24, use_returns=False, title=f"BACEN Series {code}")
    st.pyplot(fig3, width="stretch")

    fig4, metrics_diff = plot_acf_pacf(df, lags=24, use_returns=True, title=f"BACEN Series {code}")
    st.pyplot(fig4, width="stretch")

     # -------- AI INTERPRETATION (RAW SERIES) --------

    prompt_llm_lags = f"""
    You are an econometric analyst interpreting ACF and PACF plots for a BACEN macroeconomic time series.
    Use the provided diagnostics to make concise, model-specific conclusions — not generic explanations.

    **Diagnostics Summary**

    Raw (Level) Series:
    - Significant ACF lags: {metrics_raw['significant_acf_lags']}
    - Significant PACF lags: {metrics_raw['significant_pacf_lags']}

    Differenced (Returns) Series:
    - Significant ACF lags: {metrics_diff['significant_acf_lags']}
    - Significant PACF lags: {metrics_diff['significant_pacf_lags']}
    
    **Guidance for reasoning**
    - If the PACF of the raw series cuts off after lag 2, suggest an AR(2) process for the level data.
    - If the differenced series shows no significant autocorrelation confirm that differencing achieved stationarity (d=1).
    - Suggest the appropriate ARIMA(p,d,q) specification.

    **Output format**
    Write a short interpretation covering the appropriate ARIMA(p,d,q) specification.
    """

    if st.button("🧠 Generate Interpretation (Lags)", width="stretch", key="ai_lags"):
        with st.spinner("Generating AI interpretation..."):
            try:
                ai_text = generate_prompt_from_data(prompt_llm_lags)
                st.session_state.ai_text_lags = ai_text
            except Exception as e:
                st.error(f"⚠️ Error generating AI interpretation: {e}")

    if st.session_state.ai_text_lags:
        st.info(st.session_state.ai_text_lags)

    # -------------------------------------------------
    st.markdown('<div class="dashed-line"></div>', unsafe_allow_html=True)
    st.subheader("Compute Baseline  Predictions/Forecasting ")

    model_type = st.selectbox(
        "Select baseline model type:",
        ["single", "double", "holtwinters"],
        format_func=lambda x: x.title()
    )

    col1, col2, col3 = st.columns(3)
    alpha = col1.number_input("α (level smoothing)", 0.0, 1.0, value=0.3)
    beta = col2.number_input("β (trend smoothing)", 0.0, 1.0, value=0.1)
    gamma = col3.number_input("γ (seasonal smoothing)", 0.0, 1.0, value=0.1)

    steps_ahead = st.number_input("Forecast horizon (steps ahead)", 1, 12, value=2)
    season_periods = st.number_input("Seasonal period (Only holtwinters)", 1, 52, value=12)

    if st.button("🚀 Run Forecast (Rolling Evaluation)", use_container_width=True):
        # Run model
        result = baseline_forecast(df, model_type="double", alpha=0.3, beta=0.1, steps_ahead=2)

        # Combine actuals + forecasts
        df_proc = result["df_processed"]
        combined = pd.concat(
            [
                df_proc[["Value"]].rename(columns={"Value": "Actual"}),
                result["rolling_forecast"].rename("Forecast")
            ],
            axis=1
        )
        print(combined)
        fig = px.line(
            combined,
            x=combined.index,
            y=["Actual", "Forecast"],
            title=f"Rolling Forecast vs Actuals — MAPE: {result['mape']:.2f}% | RMSE: {result['rmse']:.2f}",
            markers=True
        )
        fig.update_layout(legend_title_text="Series")
        st.plotly_chart(fig, use_container_width=True)

        # Display next 2-period forecast
        st.write("**Next 2-step Forecast:**")
        st.dataframe(result["forecast_out"])