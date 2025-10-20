# AI BACEN Agent SGS Time Series Web App
import warnings
from datetime import datetime

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.bacen_agent_utils import (bacen_agent_load_langchain,
                                   bacen_agent_plot_full,
                                   bacen_agent_plot_series, plot_acf_pacf,
                                   time_series_diagnostics)

warnings.filterwarnings("ignore")


@st.cache_resource(show_spinner="Loading SentenceTransformer model...")
def load_embedder():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


embedder = load_embedder()


# Custom CSS
st.markdown(
    """
    <style>
        .dashed-line {
            border-top: 2px dashed #1E90FF; /* blue line */
            margin-top: 6px;   /* closer to the subheader */
            margin-bottom: 12px; /* small space before content */
        }
    </style>
""",
    unsafe_allow_html=True,
)


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


# Run text
if st.button("🔍 Run Analysis"):
    if not prompt.strip():
        st.warning("Please enter a prompt")
    else:
        with st.spinner("Querying BACEN API..."):
            df, code, best_key, sim = bacen_agent_load_langchain(
                prompt, start=start_date, embedder=embedder
            )

        if df.empty:
            st.error("⚠️ No results found for that prompt.")
        else:
            st.success(f"Retrieved {len(df)} rows for code {code}")

            # Display data preview
            st.markdown('<div class="dashed-line"></div>', unsafe_allow_html=True)
            st.subheader("Summary Statistics (Raw Series)")
            col1, col2 = st.columns(2)
            with col1:
                fig1 = bacen_agent_plot_series(df, code, use_returns=False)
                st.pyplot(fig1, width="stretch", bbox_inches="tight")

            with col2:
                tab_raw = time_series_diagnostics(df, lags=12, use_returns=False)
                selected_rows = tab_raw[
                    tab_raw["Statistic"].isin(
                        [
                            "Mean",
                            "Standard deviation",
                            "Skewness",
                            "Kurtosis",
                            "Jarque-Bera (p)",
                            "ADF (p)",
                        ]
                    )
                ]
                st.dataframe(selected_rows, width="stretch", hide_index=True)

            st.markdown('<div class="dashed-line"></div>', unsafe_allow_html=True)
            st.subheader("Summary Statistics (Diff Series)")
            col1, col2 = st.columns(2)
            with col1:
                fig_diff = bacen_agent_plot_series(df, code, use_returns=True)
                st.pyplot(fig_diff, width="stretch", bbox_inches="tight")

            with col2:
                tab_diff = time_series_diagnostics(df, lags=12, use_returns=True)
                selected_rows = tab_diff[
                    tab_diff["Statistic"].isin(
                        [
                            "Mean",
                            "Standard deviation",
                            "Skewness",
                            "Kurtosis",
                            "Jarque-Bera (p)",
                            "ADF (p)",
                            "Ljung-Box Q-test (p)",
                            "ARCH test (p)",
                        ]
                    )
                ]
                st.dataframe(selected_rows, width="stretch", hide_index=True)

            st.markdown('<div class="dashed-line"></div>', unsafe_allow_html=True)
            st.subheader("Lag Definitions - ACF and PACF plots")

            fig3 = plot_acf_pacf(
                df, lags=24, use_returns=False, title=f"BACEN Series {code}"
            )
            st.pyplot(fig3, width="stretch", bbox_inches="tight")

            fig4 = plot_acf_pacf(
                df, lags=24, use_returns=True, title=f"BACEN Series {code}"
            )
            st.pyplot(fig4, width="stretch", bbox_inches="tight")
