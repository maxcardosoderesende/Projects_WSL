
# AI BACEN Agent SGS Time Series Web App

import streamlit as st
import pandas as pd
from datetime import datetime
from src.bacen_agent_utils import (
    bacen_agent_load_langchain,
    bacen_agent_plot,
    time_series_diagnostics,
    plot_acf_pacf
)

import warnings
warnings.filterwarnings("ignore")

# -------------------------------
#  Streamlit UI
# -------------------------------
st.set_page_config(page_title="BACEN AI time-seriesAgent", layout="wide")
st.title("BACEN SGS Time-series analysis assistant")

st.markdown(
    """
    This app lets you query **BACEN (SGS)** time series using natural language prompts.
    Example:
    - *“Expectativa de IPCA focus média”*
    - *“Saldo poupança SBPE”*
    - *“Taxa Selic anualizada”*
    """
)

# Prompt input
prompt = st.text_input("🗣️ Enter your prompt:", placeholder="e.g., saldo poupanca SBPE")

# Date range
col1, col2 = st.columns(2)
start_date = col1.date_input("Start date", datetime(2018, 1, 1))
end_date = col2.date_input("End date", datetime.today())

# Run button
if st.button("🔍 Run Analysis"):
    if not prompt.strip():
        st.warning("Please enter a prompt")
    else:
        with st.spinner("Querying BACEN API..."):
            df, code, best_key, sim  = bacen_agent_load_langchain(prompt, start=start_date)

        if df.empty:
            st.error("⚠️ No results found for that prompt.")
        else:
            st.success(f"Retrieved {len(df)} rows for code {code}")

            # Display data preview
            # vst.dataframe(df.head())

            # Plot
            fig = bacen_agent_plot(df, code, prompt)
            if fig:
                st.pyplot(fig, width='stretch')

            # Time-series diagnostics differenced data
            st.subheader("📊 Time-series diagnostics - Differenced data")
            diag = time_series_diagnostics(df, use_returns=True, lags=12, title=f"BACEN Series {code}")
            st.dataframe(diag)

            col1, col2 = st.columns(2)

            with col1:
                fig1 = plot_acf_pacf(df, lags=24, use_returns=False, title=f"BACEN Series {code}")
                st.pyplot(fig1, use_container_width=True, bbox_inches="tight")

            with col2:
                fig2 = plot_acf_pacf(df, lags=24, use_returns=True, title=f"BACEN Series {code}")
                st.pyplot(fig2, use_container_width=True, bbox_inches="tight")
