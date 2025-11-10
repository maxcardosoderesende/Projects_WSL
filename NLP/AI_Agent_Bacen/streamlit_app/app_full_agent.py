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

from openai import OpenAI
import seaborn as sns

# AWS-Chromos
from chronos import ChronosPipeline

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Get openai API Keys




from dotenv import load_dotenv
import os

# Load .env from same folder (streamlit_app)
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)


# Create client with explicit API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


from src.bacen_agent_utils import (
    bacen_agent_load_similarity_openai,
    bacen_agent_plot_series,
    plot_acf_pacf,
    time_series_diagnostics,
    plot_residual_diagnostics,
    baseline_forecast,
    prophet_forecast_standard,
    prophet_forecast_standard_expanding_window,
    aws_chromos,
    run_arima
)


import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)


def generate_prompt_from_data(prompt_text: str) -> str:
    """
    Uses OpenAI GPT-4o-mini to generate natural-language insights
    or summaries from a provided text prompt.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a Data Sceintist specialized in Brazilian Central Bank "
                    "(BACEN/SGS) time-series interpretation. "
                    "Write clear, structured insights for model evaluation "
                    "and economic interpretation."
                )},
                {"role": "user", "content": prompt_text},
            ],
            max_tokens=400,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI API error: {e}"



def plot_best_model(result, model_name: str):
    """
    Plot Actual, Rolling, and Out-of-sample forecasts for the best model.
    Handles Prophet, ARIMA, and Chronos schemas automatically.
    """
    # --- Detect structure ---
    df_hist = result.get("df_processed", pd.DataFrame())
    metrics = result.get("metrics", {})
    model_type = model_name.lower()

    # --- Identify forecast dataframe based on model type ---
    if "forecast_df" in result:
        df_forecast = result["forecast_df"]                # ARIMA
    elif "results_df_aws" in result:
        df_forecast = result["results_df_aws"]             # Chronos
    elif "forecast_out" in result:
        df_forecast = result["forecast_out"]               # Prophet
        df_forecast["type"] = "out_of_sample"              # synthetic column for compatibility
    else:
        df_forecast = pd.DataFrame()

    # --- Rename columns to unify schema ---
    rename_map = {
        "timestamp": "Date",
        "target": "Value",
        "forecast_value": "Forecast",
        "true_value": "Actual"
    }
    df_hist = df_hist.rename(columns={k: v for k, v in rename_map.items() if k in df_hist.columns})
    df_forecast = df_forecast.rename(columns={k: v for k, v in rename_map.items() if k in df_forecast.columns})

    # --- Ensure proper datetime parsing ---
    for col in ["Date"]:
        if col in df_hist.columns:
            df_hist[col] = pd.to_datetime(df_hist[col], errors="coerce")
        if col in df_forecast.columns:
            df_forecast[col] = pd.to_datetime(df_forecast[col], errors="coerce")

    # --- Split forecast into rolling vs out-of-sample ---
    if "type" in df_forecast.columns:
        pred_forecast = df_forecast[df_forecast["type"] == "rolling_eval"]
        out_forecast = df_forecast[df_forecast["type"] == "out_of_sample"]
    else:
        pred_forecast = pd.DataFrame()
        out_forecast = df_forecast.copy()

    # --- Build Plotly Figure ---
    fig = go.Figure()

    # Actual (historical)
    if not df_hist.empty and "Value" in df_hist.columns:
        fig.add_trace(go.Scatter(
            x=df_hist["Date"], y=df_hist["Value"],
            mode="lines", name="Actual", line=dict(color="lightblue", width=2)
        ))

    # Rolling Forecast (in-sample predictions)
    if not pred_forecast.empty and "Forecast" in pred_forecast.columns:
        fig.add_trace(go.Scatter(
            x=pred_forecast["Date"], y=pred_forecast["Forecast"],
            mode="lines", name="Forecast (Rolling)",
            line=dict(color="orange", dash="dot", width=2)
        ))

    # Out-of-sample Forecast
    if not out_forecast.empty and "Forecast" in out_forecast.columns:
        fig.add_trace(go.Scatter(
            x=out_forecast["Date"], y=out_forecast["Forecast"],
            mode="lines", name="Forecast (Out-of-sample)",
            line=dict(color="green", dash="dot", width=2)
        ))

    # --- Add shaded forecast region ---
    if not df_hist.empty and (not out_forecast.empty or not pred_forecast.empty):
        last_train_date = df_hist["Date"].max()
        last_forecast_date = df_forecast["Date"].max()
        if pd.notna(last_train_date) and pd.notna(last_forecast_date):
            fig.add_vrect(
                x0=last_train_date, x1=last_forecast_date,
                fillcolor="rgba(100,150,255,0.1)", opacity=0.3,
                layer="below", line_width=0,
                annotation_text="Forecast Zone", annotation_position="top left"
            )

    # --- Title and Layout ---
    title_text = f"<b>{model_name} Forecast</b><br>"
    if metrics:
        metric_parts = [f"{k}: {v:.2f}" for k, v in metrics.items()]
        title_text += " | ".join(metric_parts)

    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
        height=420
    )

    return fig




def run_full_ai_agent(prompt: str, start_date, end_date):
    """
    Run the full AI time-series pipeline from prompt to forecast and summary.
    """
    try:
        # 1️Data Retrieval
        df, code, best_key, sim = bacen_agent_load_similarity_openai(prompt, start=start_date, client=client)

        if df is None or df.empty:
            return {"error": "No data returned."}

        # 2️⃣ Pre-Tests
        fig_series_levels = bacen_agent_plot_series(df, code, use_returns=False)
        fig_series_returns = bacen_agent_plot_series(df, code, use_returns=True)
        tab_raw = time_series_diagnostics(df, lags=12, use_returns=False)
        tab_diff = time_series_diagnostics(df, lags=12, use_returns=True)
        acf_fig, metrics_raw = plot_acf_pacf(df, lags=24, use_returns=False)
        pacf_fig, metrics_diff = plot_acf_pacf(df, lags=24, use_returns=True)

        # 3️⃣ Models
        baseline = baseline_forecast(df, steps_ahead=30)
        arima = run_arima(df, steps_ahead=30)
        prophet_traditional = prophet_forecast_standard(df, steps_ahead=30)
        prophet_window = prophet_forecast_standard_expanding_window(df, steps_ahead = 30)
        chronos = aws_chromos(df, pipeline=ChronosPipeline.from_pretrained("amazon/chronos-t5-tiny"))

        # 4️⃣ Model performance summary
        summary = pd.DataFrame([
            {"Model": "Baseline", **baseline["metrics"]},
            {"Model": "ARIMA", **arima["metrics"]},
            {"Model": "Prophet_traditional", "MAE": prophet_traditional["mae"], "MAPE": prophet_traditional["mape"]},
            {"Model": "Prophet_window", "MAE": prophet_window["mae"], "MAPE": prophet_window["mape"]},
            {"Model": "Chronos", **chronos["metrics"]}
        ])

        # Determine best model based on MAPE
        best_model_name = summary.loc[summary["MAPE"].idxmin(), "Model"]

        model_results = {
            "Baseline": baseline,
            "ARIMA": arima,
            "Prophet_traditional": prophet_traditional,
            "Prophet_window": prophet_window,
            "Chronos": chronos
        }

        best_model_result = model_results[best_model_name]

        summary_prompt = f"""
        You are an Time-series AI Engineert interpreting results from multiple forecasting models
        applied to a Brazilian Central Bank (BACEN) time series.

        Each model has different assumptions about stationarity differencing and how it is implemented.

        Given the model performance summary:
        {summary.to_dict(orient='records')}

        Explain clearly and concisely:
        1️⃣ How ARIMA, Prophet, AWS Chronos, and Exponential Smoothing differ in terms of:
        - Treatment of trend and stationarity
        - Need for differencing or not
        - Whether they assume linearity
        2️⃣ Which model likely performed best and why, given the MAPE.
        3️⃣ Which model is more robust to non-stationary data.
        4️⃣ One economic insight about the series trend or volatility.
   
        Return a clear, structured paragraph — concise, yet analytical.
        """


        insights = generate_prompt_from_data(summary_prompt)

        return {
            "data": df,
            "plot_series_levels": fig_series_levels,
            "plot_series_returns": fig_series_returns,
            "tab_raw": tab_raw,
            "tab_diff": tab_diff,
            "acf_fig": acf_fig,
            "pacf_fig": pacf_fig,
            "results": {
                "baseline": baseline,
                "arima": arima,
                "prophet_traditional": prophet_traditional,
                "chronos": chronos
            },
            "metrics_summary": summary,
            "best_model_name": best_model_name,
            "best_model_result": best_model_result,
            "ai_insights": insights
        }

    except Exception as e:
        return {"error": str(e)}


# -------------------------------------------------
# User inputs for the AI Agent
# -------------------------------------------------
st.set_page_config(page_title="BACEN AI Time-Series Agent", layout="wide")
st.title("Time-Series Analysis Assistant")

st.markdown(
    """
    This app lets you query **BACEN (SGS)** time-series data using openAI natural-language prompts.
    Example queries:
    - *“Expectativa de IPCA Focus média - IPCA inflation rate”*
    - *“Saldo poupança SBPE - Total savings”*
    - *“Taxa Selic anualizada - Annualized interest rate 252 days”*
    """
)

prompt = st.text_input(
    "Enter the time-series you want to analyze:",
    placeholder="e.g., Saldo poupança SBPE or Taxa Selic"
)

col1, col2 = st.columns(2)
start_date = col1.date_input("Start date", datetime(2018, 1, 1))
end_date = col2.date_input("End date", datetime.today())


st.markdown("---")
st.subheader("🤖 Run Full AI Agent Analysis")

if st.button("Run Full AI Time-Series Agent",  width="stretch"):
    with st.spinner("Agent running full analysis..."):
        result = run_full_ai_agent(prompt, start_date, end_date)
        st.session_state.full_agent_result = result

if "full_agent_result" in st.session_state:
    res = st.session_state.full_agent_result
    ## === Two-column layout ===
    col1, col2 = st.columns(2)
    st.markdown("#### Requested BACEN Time Series")
    with col1:
        st.pyplot(res["plot_series_levels"], width='stretch')
    with col2:
        st.pyplot(res["plot_series_returns"], width='stretch')

    if "error" in res:
        st.error(res["error"])
    else:
        st.success("✅ Agent completed full pipeline!")
        st.dataframe(res["metrics_summary"], hide_index=True)

        # ---- Best Model Plot ----
        st.markdown(f"### Best Model: **{res['best_model_name']}**")
        best_fig = plot_best_model(res["best_model_result"], res["best_model_name"])
        st.plotly_chart(best_fig, use_container_width=True)

        # ---- AI Text Summary ----
        st.markdown("#### AI Summary")
        st.info(res["ai_insights"])