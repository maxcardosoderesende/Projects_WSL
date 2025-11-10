# 🤖 AI Time-Series Agent v1.1

The **AI Time-Series Agent** acts as an autonomous *Data Scientist*, capable of understanding, diagnosing, forecasting, and interpreting economic time-series data — initially built for **Brazilian Central Bank (BACEN/SGS)** indicators.

This intelligent web app unifies **classical econometric models**, **modern deep-learning forecasters**, and **LLM-powered reasoning** into one automated analytical pipeline — delivering not only forecasts but **context-aware AI explanations** and **model recommendations**.

Built with **Python + Streamlit**, it orchestrates the full time-series lifecycle — from **data ingestion and pre-testing** to **forecast generation** and **AI-driven interpretation**.

---

## ⚙️ Workflow Overview

1. **🔍 Data Retrieval** – Connects to BACEN’s SGS API to automatically load and structure time-series data.  
2. **📊 Statistical Diagnostics** – Applies ADF, Skewness, Kurtosis, Jarque–Bera, Ljung–Box, and ARCH tests to assess stationarity and volatility.  
3. **📈 ACF/PACF Visualization** – Displays correlation structures to guide ARIMA specification.  
4. **📉 Forecasting Layer** – Runs a unified model stack:  
   - **Statistical:** ARIMA, Holt-Winters, Exponential Smoothing  
   - **Machine Learning:** Prophet (standard + expanding window)  
   - **Pre-trained Deep Model:** AWS **Chronos-T5-tiny** (transformer-based time-series forecaster)  
5. **🧩 Generative Reasoning Layer (OpenAI GPT-4o-mini)** –  
   - Embeds user queries  
   - Selects the most relevant BACEN indicator  
   - Generates **natural-language insights** explaining diagnostics, model differences, and trend interpretation  

---

## 🏗️ Architecture

| Layer | Description | Technology Stack |
|-------|--------------|------------------|
| **Data Layer** | Fetches and structures BACEN SGS data or local CSV inputs | `pandas`, `requests` |
| **Diagnostics Layer** | Runs statistical tests for normality, autocorrelation, and heteroskedasticity | `statsmodels`, `scipy` |
| **Model Layer** | Trains baseline & advanced models; orchestrates rolling and expanding forecasts | `ARIMA`, `Prophet`, `ExponentialSmoothing`, `AWS-Chronos (ChronosPipeline)` |
| **AI Reasoning Layer** | Uses **OpenAI embeddings + GPT-4o-mini** to interpret diagnostics, suggest models, and summarize findings | `openai`, `tiktoken` |
| **Visualization Layer** | Interactive ACF/PACF, diagnostics, and forecast plots | `matplotlib`, `plotly`, `streamlit` |
| **Interface Layer** | User-friendly app for input prompts and dynamic exploration | `Streamlit`, `SessionState` |

---

## 🧮 Model Strategy

The system implements a **multi-paradigm forecasting strategy**:

- **Statistical models (ARIMA, ES):** Capture linear and trend-stationary structures.  
- **Prophet (additive model):** Flexible for non-stationary trends and seasonalities.  
- **AWS Chronos (pre-trained transformer):** Zero-shot generalization to unseen time-series distributions — eliminating manual differencing.  
- **OpenAI GPT-4o-mini (LLM Reasoning):** Translates statistical complexity into clear, economic interpretations.  
  Uses **embeddings search** to map user prompts (in Portuguese or English) to the correct BACEN indicator.

This hybrid orchestration enables the agent to **autonomously select**, **train**, and **interpret** the best model for a given time-series without explicit user configuration.

---

## 🤖 AI Agent Logic

The agent performs:
- Automatic **feature interpretation** (trend, seasonality, volatility)  
- Model **selection and evaluation** using MAPE and MAE  
- Generation of **economic context insights** (e.g., exchange rate shifts, inflation, reserves)  
- **Natural-language summaries** powered by GPT-4o-mini  

---

### 💡 Example Output

> “The PACF of the raw series indicates an AR(2) structure.  
> After first differencing, the ADF test rejects the null (p < 0.05), confirming stationarity.  
> Among evaluated models, ARIMA(2,1,0) provides the best MAPE balance.  
> Chronos exhibits superior performance under non-linear patterns due to its pre-trained transformer backbone.”  

---

## ☁️ Deployment

Deployed both locally and on **AWS (SageMaker + EC2 - Caution with EC2 costs)**.  

The framework supports seamless extension to other APIs or private datasets, showcasing a **next-generation AI-assisted time-series lab** — combining human reasoning, statistical rigor, and foundation-model intelligence.

---



---

## 🧰 Tech Stack Summary

`Python` • `Streamlit` • `Plotly` • `statsmodels` • `Prophet` • `OpenAI` • `ChronosPipeline` • `AWS SageMaker` • `EC2` • `pandas`

---

## 🚀 Future Extensions
- Integrate Nixtla’s NeuralForecast (N-BEATSx, TFT) for advanced multi-horizon forecasting
- Add multivariate forecasting with macroeconomic covariates (e.g. GDP, interest rates, exchange rates)
- Implement RAG (Retrieval-Augmented Generation) to dynamically fetch and inject contextual variables from economic databases or news feeds into model pipelines
- Automate anomaly detection and changepoint detection (CUSUM, Bayesian online)
- Deploy fully on Streamlit Cloud or AWS Elastic Beanstalk for public demo
- Expand to international datasets (FRED, ECB, IMF, World Bank, OECD)

---

## 🧑‍💻 Author

**Max Cardoso de Resende**  
📧 [maxscardosoderesende@gmail.com](mailto:maxscardosoderesende@gmail.com)  
🔗 [linkedin.com/in/maxresende](https://www.linkedin.com/in/max-resende-006757b0/)