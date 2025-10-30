# AI BACEN Agent SGS Time Series Web App
import textwrap
import warnings
from datetime import datetime
import os

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px 
import plotly.graph_objects as go
from dotenv import load_dotenv
from openai import OpenAI
import seaborn as sns

from src.bacen_agent_utils import (
    bacen_agent_load_similarity_openai,
    bacen_agent_plot_series,
    plot_acf_pacf,
    time_series_diagnostics,
    plot_residual_diagnostics,
    baseline_forecast,
    prophet_forecast_standard,
    prophet_forecast_standard_expanding_window
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
for key in ["df", "code", "tab_raw", "tab_diff", "ai_text_raw", "ai_text_diff", "ai_text_lags",
             "baseline_result", "prophet_result_mean", "prophet_result_traditional", 'prophet_result_window']:
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
            st.success(f"Retrieved {len(df)} rows for code {code} | {best_key} ")
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


# -------------------------------------------------------------------
# FORECASTING SESSION
# -------------------------------------------------------------------
st.markdown('<div class="dashed-line"></div>', unsafe_allow_html=True)
st.subheader("Forecasting Models")

if st.session_state.df is not None:
    df = st.session_state.df

# === Two-column layout ===
col1, col2, col3 = st.columns(3)

# -------------------------------------------------------------------
# BASELINE MODEL (EXPONENTIAL SMOOTHING)
# -------------------------------------------------------------------
with col1:
    st.markdown("#### BASELINE: Exponential Smoothing")

    model_type = st.selectbox(
        "Select model type:",
        ["single", "double", "holtwinters"],
        index=1,
        format_func=lambda x: x.title()
    )

    c1, c2, c3, c4 = st.columns(4)
    # single -> Data is relatively stable with no trend or repeating pattern
    # α close to 1 → reacts quickly to recent changes.
    alpha = c1.number_input("α (level)", 0.0, 1.0, 0.3, 0.05)
    # Doble -> Level + Trend -> Data has a consistent upward or downward trend but no seasonality
    # Adds a trend term that evolves over time.
    beta = c2.number_input("β (trend)", 0.0, 1.0, 0.1, 0.05)
    # Hot Winters -> Data has both a trend and repeating seasonal patterns (e.g. weekly, monthly)
    # α (level), β (trend), γ (seasonal), and season_periods (e.g. 12 for months)
    gamma = c3.number_input("γ (seasonal)", 0.0, 1.0, 0.1, 0.05)
    season_periods = c4.number_input("Seasonal period", 1, 52, 12)
    # steps_ahead_base = c5.number_input("Forecast horizon (steps ahead)", 1, 12, n)
    steps_ahead = 30

    if st.button("🚀 Run Baseline Forecast", use_container_width=True):
        with st.spinner("Running Baseline Forecast..."):
            try:
                result_baseline = baseline_forecast(
                    df,
                    model_type=model_type,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    season_periods=season_periods,
                    steps_ahead = steps_ahead
                )
                # ✅ Persist results
                st.session_state.baseline_result = result_baseline

            except Exception as e:
                st.error(f"⚠️ Error in baseline forecast: {e}")

    # ✅ Show previous results (if exist)
    if st.session_state.baseline_result is not None:
        result = st.session_state.baseline_result
        df_proc = result["df_processed"]

        # ---- Chart ----
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_proc.index, y=df_proc["Value"],
                                 mode="lines", name="Actual", line=dict(color="lightblue", width=2)))
        fig.add_trace(go.Scatter(x=result["rolling_forecast"].index, y=result["rolling_forecast"],
                                 mode="lines", name="Forecast (rolling)",
                                 line=dict(color="orange", dash="dot", width=2)))
        fig.add_trace(go.Scatter(x=result["forecast_out"]["Date"],y=result["forecast_out"]["Forecast"],
                                mode="lines", name="Forecast (Future)",
                                line=dict(color="green", dash="dot", width=2)))
        fig.update_layout(
            title=f"<b>{model_type.title()} Rolling Forecast</b><br>"
                  f"MAPE: {result['mape']:.2f}% | RMSE: {result['rmse']:.2f}",
            xaxis_title="Date",
            yaxis_title="Value",
            legend=dict(x=0.01, y=0.99),
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Next Forecasts")
        # --- Format the next forecast dataframe safely ---

        forecast_out = result["forecast_out"].copy()
        forecast_out["Date"] = pd.to_datetime(forecast_out["Date"]).dt.strftime("%Y-%m-%d")
        forecast_out["Forecast"] = forecast_out["Forecast"].astype(float)

        st.dataframe(forecast_out.style.format({"Forecast": "{:,.2f}"}),
                     width='stretch', hide_index=True)


        st.markdown('#### Error Diagnostics')
        residuals = result["residuals"]
        fig_resid = plot_residual_diagnostics(residuals)
        st.pyplot(fig_resid, width='stretch')

# -------------------------------------------------------------------
# PROPHET MODEL Traditional (non-rolling)
# -------------------------------------------------------------------

with col2:
    st.markdown("#### Prophet Forecasting - Traditional")

    seasonality_type_std = st.selectbox(
        "Select seasonality type:",
        ["None", "additive", "multiplicative"],
        index=1,
        format_func=lambda x: x.title(),
        key="seasonality_type_traditional"
    )

    # --- Seasonality selectors (each must have a unique key)
    c1, c2, c3 = st.columns(3)
    weekly_seasonality_std = c1.selectbox(
        "Weekly Seasonality", [True, False], index=1, key="weekly_traditional"
    )
    yearly_seasonality_std = c2.selectbox(
        "Yearly Seasonality", [True, False], index=1, key="yearly_traditional"
    )
    steps_ahead_std = c3.number_input(
        "Forecast horizon (steps ahead):",
        min_value=1, max_value=90, value=30, step=1,
        key="steps_ahead_std_window"
    )

    if st.button("Run Traditional Prophet Forecast", use_container_width=True, key="run_prophet_traditional"):
        with st.spinner("Running standard Prophet forecast..."):
            try:
                result_prophet_std = prophet_forecast_standard(
                    df,
                    test_size=0.2,
                    steps_ahead=steps_ahead_std,
                    weekly_seasonality=weekly_seasonality_std,
                    yearly_seasonality=yearly_seasonality_std,
                    seasonality_mode=seasonality_type_std
                )

                st.session_state.prophet_result_traditional = result_prophet_std

            except Exception as e:
                st.error(f"⚠️ Error running Prophet forecast: {e}")


    # --- Display results if available ---
    if st.session_state.prophet_result_traditional is not None:
        result_prophet_std = st.session_state.prophet_result_traditional
        df_proc_p = result_prophet_std["df_processed"]
    
        # --- Plot ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_proc_p["Date"], y=df_proc_p["Value"],
            mode="lines", name="Actual", line=dict(color="lightblue", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=result_prophet_std["test_forecast"].index,
            y=result_prophet_std["test_forecast"].values,
            mode="lines", name="Forecast (Test)",
            line=dict(color="orange", dash="dot", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=result_prophet_std["forecast_out"]["Date"],
            y=result_prophet_std["forecast_out"]["Forecast"],
            mode="lines", name="Forecast (Future)",
            line=dict(color="green", dash="dot", width=2)
        ))
        fig.update_layout(
            title=f"<b>Prophet Traditional Forecast</b><br>"
                  f"MAPE: {result_prophet_std['mape']:.2f}% | RMSE: {result_prophet_std['rmse']:.2f}",
            xaxis_title="Date", yaxis_title="Value",
            legend=dict(x=0.01, y=0.99), template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Next Forecasts")
        forecast_out = result_prophet_std["forecast_out"].copy()
        forecast_out["Date"] = pd.to_datetime(forecast_out["Date"]).dt.strftime("%Y-%m-%d")
        forecast_out["Forecast"] = forecast_out["Forecast"].astype(float)

        st.dataframe(
            forecast_out.style.format({"Forecast": "{:,.2f}"}),
            use_container_width=True,
            hide_index=True
        )


        st.markdown('#### Error Diagnostics')
        residuals = result_prophet_std["residuals"]
        fig_resid = plot_residual_diagnostics(residuals)
        st.pyplot(fig_resid, width='stretch')


 #-------------------------------------------------------------------
# PROPHET MODEL Rolling Expanding Window
# -------------------------------------------------------------------

with col3:
    st.markdown("#### Prophet Forecasting - Rolling Window")

    seasonality_type_window = st.selectbox(
        "Select seasonality type:",
        ["None","additive", "multiplicative",],
        index=1,
        format_func=lambda x: x.title()
    )

    # --- Seasonality selectors ---
    c1, c2, c3 = st.columns(3)
    #daily_seasonality_roll = c1.selectbox("Daily Seasonality", [True, False], index=1)
    weekly_seasonality_roll = c1.selectbox("Weekly Seasonality", [True, False], index=1)
    yearly_seasonality_roll = c2.selectbox("Yearly Seasonality", [True, False], index=1)
    steps_ahead_base = c3.number_input(
        "Forecast horizon (steps ahead):",
        min_value=1, max_value=90, value=30, step=1,
        key="steps_ahead_window"
    )

    # --- Run Prophet Forecast ---
    if st.button("Run Prophet Forecast", use_container_width=True):
        with st.spinner("Running Prophet Forecast..."):
            try:
                result_prophet = prophet_forecast_standard_expanding_window(
                    df,
                    steps_ahead=steps_ahead_base,
                    weekly_seasonality=weekly_seasonality_roll,
                    yearly_seasonality=yearly_seasonality_roll,
                    seasonality_mode=seasonality_type_window,
                    changepoint_prior_scale = 0.8,  # or even 1.0
                    n_changepoints = 300,
                    changepoint_range = 1.0

                )
                st.session_state.prophet_result_window = result_prophet  # ✅ persist

            except Exception as e:
                st.error(f"⚠️ Error running Prophet forecast: {e}")

    # --- Display results if available ---
    if st.session_state.prophet_result_window is not None:
        result_prophet = st.session_state.prophet_result_window
        print("Result Prophet keys:", result_prophet.keys())
        df_proc_p = result_prophet["df_processed"]
        
        # --- Plot ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_proc_p["Date"], y=df_proc_p["Value"],
            mode="lines", name="Actual", line=dict(color="lightblue", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=result_prophet["rolling_forecast"].index,
            y=result_prophet["rolling_forecast"].values,
            mode="lines",
            name="Forecast (Rolling)",
            line=dict(color="orange", dash="dot", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=result_prophet["forecast_out"]["Date"],
            y=result_prophet["forecast_out"]["Forecast"],
            mode="lines", name="Forecast (Future)",
            line=dict(color="green", dash="dot", width=2)
        ))
        fig.update_layout(
            title=f"<b>Prophet Rolling Forecast</b><br>"
                f"MAPE: {result_prophet['mape']:.2f}% | RMSE: {result_prophet['rmse']:.2f}",
            xaxis_title="Date", yaxis_title="Value",
            legend=dict(x=0.01, y=0.99), template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Format the next forecast dataframe safely ---
        forecast_out = result_prophet["forecast_out"].copy()

        # Ensure Date is datetime and Forecast is numeric 1D
        forecast_out["Date"] = pd.to_datetime(forecast_out["Date"]).dt.strftime("%Y-%m-%d")
        forecast_out["Forecast"] = forecast_out["Forecast"].astype(float)

        # Create clean display dataframe+69857
        forecast_prophet_df = forecast_out[["Date", "Forecast"]].reset_index(drop=True)

        st.markdown("#### Next Forecasts")
        st.dataframe(
            forecast_prophet_df.style.format({"Forecast": "{:,.2f}"}),
            use_container_width=True,
            hide_index=True
        )

        st.markdown('#### Error Diagnostics')
        resid_exp_window = result_prophet["residuals"]
        fig_resid = plot_residual_diagnostics(resid_exp_window)
        st.pyplot(fig_resid, width='stretch')
