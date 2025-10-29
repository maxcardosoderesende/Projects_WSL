
---

## AI Time-Series Agent v1.0

The **AI Time-Series Agent** functions as a Data Scientist — a system that understands, models, and explains time-series data provided by the Brazilian Cebtral Bank indicators (for now).

I have built an interactive forecasting and analytics web app that combines **classical time-series models** (ARIMA, Holt-Winters, Exponential Smoothing) with **LLM-based reasoning** (GPT-4o-mini or LLaMA).  

Built with **Python + Streamlit**, this project automates the full time-series workflow — from **data ingestion and diagnostics** to **forecasting and AI-powered interpretation**.

## Workflow Overview:

1. **Data Retrieval:** via BACEN’s API or CSV uploads.  
2. **Statistical Diagnostics:** Augmented Dickey–Fuller (ADF), Skewness, Kurtosis, Jarque-Bera, Ljung-Box, ARCH tests.  
3. **ACF/PACF Visualization:** to identify AR/MA structures and check for stationarity.  
4. **Baseline Forecasting:** Single, Double, and Holt-Winters exponential smoothing models.  
5. **Generative AI Interpretation:** GPT-4o-mini interprets diagnostics and model outputs to produce **human-readable insights** and **model recommendations**.

## Architecture

| Layer | Description | Technology |
|-------|--------------|-------------|
| **Data Layer** | Fetches BACEN SGS data or loads CSV inputs | `pandas`, `requests` |
| **Diagnostics Layer** | Computes ADF, Skewness, Kurtosis, Ljung–Box, ARCH | `statsmodels`, `scipy` |
| **Model Layer** | Implements baseline models (Single, Double, Holt-Winters) | `SimpleExpSmoothing`, `ExponentialSmoothing` |
| **Visualization Layer** | Displays plots and metrics interactively | `matplotlib`, `plotly`, `streamlit` |
| **Reasoning Layer (AI)** | GPT-4o-mini / LLaMA interprets outputs and suggests models | `openai` |
| **Interface Layer** | User interaction, chat-style insights | `Streamlit` |


Obs: Deployed both locally and on AWS (SageMaker + EC2), this framework showcases the next generation of AI-assisted time-series modeling.

## AI Agent Logic

It:
- Interprets statistical diagnostics  
- Recommends models and parameters (e.g. ARIMA(2,1,0))  
- Explains model behavior, stationarity, and volatility  
- Translates numerical results into readable insights  

### Example Output
> “The PACF of the raw series indicates an AR(2) structure.  
> After differencing, the ADF test rejects the null (p < 0.05), confirming stationarity.  
> The recommended specification is ARIMA(2, 1, 0).”

📧 **Email:** [maxscardosoderesende@gmail.com](mailto:maxscardosoderesende@gmail.com)  
🔗 **LinkedIn:** [linkedin.com/in/maxcardoso](https://www.linkedin.com/in/max-resende-006757b0/)

