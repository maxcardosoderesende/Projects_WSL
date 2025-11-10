# BACEN AI Agent -  source functions

# Similarity
import difflib
import re
import unicodedata
import warnings
from difflib import SequenceMatcher

from tqdm import tqdm
import time

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # lib for data manipulation
import numpy as np
import requests  # access HTTP content
from bcb import sgs
from requests.adapters import HTTPAdapter, Retry
## Timne-series diagnostics tests
from scipy import stats
## langChain

from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from tqdm import trange  # lib for progress bars
from statsmodels.tsa.arima.model import ARIMA

# Prophet
from prophet import Prophet

# AWS-Chromos
from chronos import ChronosPipeline
from pandas.tseries.frequencies import to_offset
import torch

# openAI
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)


BACEN_TOP_SERIES = {
    # 🔹 External sector
    "reservas internacionais total diária": 13621,
    "transações correntes mensal saldo": 22701,
    # 🔹 Interest rates
    "taxa selic meta definida pelo copom": 432,
    "taxa selic acumulada no mês": 4390,
    "taxa selic anualizada base 252": 1178,
    "taxa básica financeira tbf": 253,
    "taxa referencial tr": 226,
    "taxa selic diária": 11,
    # 🔹 Credit and savings
    "Saldo diário de depósitos de poupança - SBPE": 239,
    # "Taxa média de depósitos de poupança (rentabilidade)": 195,
    "taxa média de juros pessoas físicas total": 20716,
    "taxa média de juros pessoas jurídicas total": 20715,
    "taxa média de juros total": 20714,
    "indicador de custo do crédito icc": 25359,
    # 🔹 Inflation indices (CPI)
    "Índice nacional de preços ao consumidor-amplo (IPCA)": 433,
    "Índice nacional de preços ao consumidor-Amplo (IPCA) - Núcleo médias aparadas com suavização": 4466,
    "ipca 15 índice nacional de preços ao consumidor amplo 15": 11428,
    "inpc índice nacional de preços ao consumidor": 188,
    # 🔹 Monetary aggregates
    "base monetária ampliada títulos do tesouro nacional carteira do mercado": 1831,
    "base monetária ampliada títulos do tesouro nacional financiamento líquido": 1832,
    "meios de pagamento ampliados operações compromissadas selic": 1839,
    "meios de pagamento ampliados títulos federais em poder do público selic": 1841,
    "dívida mobiliária participação por indexador posição de custódia": 2238,
    "dívida mobiliária participação por indexador posição em carteira": 4177,
    # 🔹 GDP and Activity
    "PIB mensal - Valores correntes em reais": 4380,
    "PIB mensal - Em dólares US$ milhões":  4385,
    "pib nominal preço de mercado trimestral": 4382,
    "ibc br índice de atividade econômica do banco central mensal dessazonalizado": 24364,
    "ibc br índice de atividade econômica do banco central mensal original": 24363,
    # 🔹 Inflation expectations (Focus survey)
    "expectativa ipca 12 meses focus média": 4333,
    # 🔹 Exchange rates
    "câmbio comercial venda r$/us$ média": 10813,
    "câmbio efetivo real índice jan 2005 = 100": 11752,
    "câmbio nominal índice jan 2005 = 100": 11753,
}


def normalize_text(text):
    """
    Normalize text for robust fuzzy matching:
    - Lowercase
    - Remove accents
    - Remove punctuation and symbols
    - Collapse multiple spaces
    """
    if not isinstance(text, str):
        return ""
    # Lowercase and strip
    text = text.lower().strip()

    # Remove accents (normalize to NFD and strip combining marks)
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )

    # Remove punctuation, symbols, and non-alphanumeric chars (keep spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# get the BACEN CODE from a text prompt
def find_bacen_code(prompt, dictionary=BACEN_TOP_SERIES, cutoff=0.3, verbose=True):
    """Find the BACEN code by fuzzy best match or partial text match."""

    # Normalize prompt and dictionary keys
    q = normalize_text(prompt)
    normalized_dict = {normalize_text(k): v for k, v in dictionary.items()}

    # Fuzzy best match
    match = difflib.get_close_matches(q, normalized_dict.keys(), n=1, cutoff=cutoff)
    if match:
        best_key = match[0]
        code = normalized_dict[best_key]
        if verbose:
            print(f"Fuzzy match: '{best_key}' → code {code}")
        return code

    # Partial substring match
    partial_matches = {k: v for k, v in normalized_dict.items() if q in k}
    if partial_matches:
        best_key = max(partial_matches, key=len)
        best_code = partial_matches[best_key]
        if verbose:
            print(f"Partial match: '{best_key}' → code {best_code}")
        return best_code

    ## No match
    print("No match found.")
    return None


def bacen_agent_load(
    prompt,
    dictionary=BACEN_TOP_SERIES,
    start=None,
    end=pd.Timestamp.now().normalize(),
    cutoff=0.3,
):
    """
    Understands the prompt, finds the right BACEN code using difflib methodology,
    lists all possible similar matches,
    and loads the data for the best match (highest similarity score).
    """
    print(f"\nProcessing prompt: '{prompt}'")

    # --- Tokenize and normalize ---
    text = normalize_text(prompt)
    tokens = text.split()
    print(f"Extracted tokens: {tokens}")

    # --- Normalize dictionary ---
    normalized_dict = {normalize_text(k): v for k, v in dictionary.items()}

    # --- Filter dictionary to keys that contain any query token ---
    filtered_items = {
        k: v for k, v in normalized_dict.items() if any(tok in k for tok in tokens)
    }

    if not filtered_items:
        print("⚠️ No partial matches found based on tokens.")
        return pd.DataFrame(), None

    # --- Compute similarity only within filtered subset ---
    scored = []
    for key, code in filtered_items.items():
        sim = SequenceMatcher(None, text, key).ratio()
        if sim >= cutoff:
            scored.append((key, code, sim))

    if not scored:
        print(f"⚠️ No matches above cutoff {cutoff}.")
        return pd.DataFrame(), None

    scored.sort(key=lambda x: (-x[2], x[1]))

    # --- Display ---
    print(f"\n🔍 Possible matches (similarity ≥ {cutoff}):")
    for key, code, sim in scored:
        print(f"  • {key:<80} → code {code:<6} (similarity={sim:.2f})")

    best_key, best_code, best_sim = scored[0]
    print(
        f"\nBest match selected: '{best_key}' → code {best_code} (similarity={best_sim:.2f})"
    )

    # --- Fetch data ---
    df = sgs.get(best_code, start=start, end=end)  # use numeric directly
    df = df.reset_index().rename(columns={"index": "Date", best_code: "Value"})
    print(f"Retrieved {len(df)} rows for code {best_code}.")
    return df, best_code


# -------------------------------------------------
# NEW OpenAI-powered retriever
# -------------------------------------------------
def bacen_agent_load_similarity_openai(
    prompt,
    dictionary=BACEN_TOP_SERIES,
    start=None,
    end=None,
    cutoff=0.35,
    client=None,
    embedding_model="text-embedding-3-small",
):
    """
    Intelligent BACEN data retriever using OpenAI embeddings.

    1. Uses OpenAI embeddings to compute semantic similarity between user prompt and BACEN dictionary.
    2. Automatically retrieves data for the best-matched series.
    3. Returns df, code, best_key, and similarity score.
    """

    if start is None:
        start = "2020-01-01"

    if end is None:
        end = pd.Timestamp.now().normalize()

    if client is None:
        client = OpenAI()  # assumes OPENAI_API_KEY in env

    # --- Step 1: Vectorize dictionary keys ---
    bacen_keys = list(dictionary.keys())
    bacen_codes = list(dictionary.values())

    # Get embeddings for all BACEN series names
    bacen_embs = []
    for i in range(0, len(bacen_keys), 100):  # batch in case you have many
        batch = bacen_keys[i:i+100]
        resp = client.embeddings.create(model=embedding_model, input=batch)
        bacen_embs.extend([e.embedding for e in resp.data])

    # --- Step 2: Get embedding for user query ---
    q_emb = client.embeddings.create(model=embedding_model, input=[prompt]).data[0].embedding

    # --- Step 3: Compute cosine similarity ---
    sims = cosine_similarity([q_emb], bacen_embs)[0]
    top_idx = sims.argmax()
    best_key = bacen_keys[top_idx]
    best_code = bacen_codes[top_idx]
    best_sim = sims[top_idx]

    # --- Step 4: Fetch BACEN data ---
    try:
        df = sgs.get(best_code, start=start, end=end)
        df = df.reset_index().rename(columns={"index": "Date", best_code: "Value"})
        last_date = df["Date"].max()
        print(f"📈 Retrieved {len(df)} rows for code {best_code} | {best_key}  | Similarity: {best_sim:.2f} | Max date {last_date}")
    except Exception as e:
        print(f"❌ Error retrieving series {best_code}: {e}")
        return pd.DataFrame(), best_code, best_key, best_sim

    return df, best_code, best_key, best_sim


def bacen_agent_plot_series(df, code, use_returns=False):
    """Time-series plot."""

    if "Value" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "Value"})

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    if use_returns:
        series = df["Value"].pct_change() * 100
        series = series.dropna()
        label = "Returns – Percentage (Δy/yₜ₋₁ * 100)"
    else:
        series = df["Value"]
        label = "Levels – Raw Series"

    # --- Plot ---
    fig = plt.figure(figsize=(10, 4))
    plt.plot(series, color="tab:blue", linewidth=1.5)
    plt.title(f" BACEN Time Series {code} – {label}", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


def bacen_agent_plot_full(df, code):
    """
    Fetch, analyze, and plot BACEN series using LangChain-based semantic matching
    """

    if df.empty:
        print("⚠️ No data to plot.")
        return

    if "Value" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "Value"})

    # --- Prepare data ---
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df_returns = df["Value"].pct_change() * 100

    # --- Detect frequency ---
    freq_days = df.index.to_series().diff().median().days
    if freq_days <= 2:
        freq_type = "daily"
        df_resampled = df["Value"].resample("W").mean()
        period = 52
    elif freq_days < 25:
        freq_type = "monthly"
        df_resampled = df["Value"].resample("M").mean()
        period = 12
    else:
        freq_type = "annual"
        df_resampled = df["Value"].copy()
        period = None

    print(
        f"Detected {freq_type} data → using resampled freq = {df_resampled.index.freqstr or 'W/M'}"
    )

    # --- Plot layout ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    fig.suptitle(f"BACEN Time  Series {code}", fontsize=12, fontweight="bold")

    axes[0, 0].plot(df.index, df["Value"], color="tab:blue", linewidth=1.5)
    axes[0, 0].set_title("Original Series")

    axes[0, 1].plot(df.index, df_returns, color="tab:orange", linewidth=1)
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Returns (Δyₜ / yₜ₋₁)")

    try:
        result = seasonal_decompose(
            df_resampled.dropna(), model="additive", period=period
        )
        axes[1, 0].plot(result.trend, color="tab:green")
        axes[1, 0].set_title("Trend")
        axes[1, 1].plot(result.seasonal, color="tab:purple")
        axes[1, 1].set_title("Seasonal (Resampled)")
    except Exception as e:
        print(f"⚠️ Decomposition failed: {e}")

    for ax in axes.flat:
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


def time_series_diagnostics(df, lags=12, use_returns=False):
    """
    Compute descriptive and diagnostic statistics for a BACEN time series DataFrame.
    Tests: Descriptive stats, ADF, Ljung-Box (Q and Q²), ARCH LM.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ['Date', 'Value'].
    lags : int
        Number of lags for autocorrelation tests.
    use_returns : bool
        If True, compute statistics on percentage returns (Δy/yₜ₋₁ * 100) and not on raw_values.
    title : str
        Optional title for printed output.
    """

    if df.empty:
        print("⚠️ No data provided.")
        return

    # --- Ensure proper column names and sorting ---
    if "Value" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "Value"})

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    # --- Use raw levels or returns ---
    if use_returns:
        df["Value"] = df["Value"].pct_change() * 100
        df = df.dropna()
        print("Using percentage returns (Δy/yₜ₋₁ * 100).")

    series = df["Value"]

    # --- Descriptive statistics ---
    n = len(series)
    mean = series.mean()
    std = series.std()
    skew = stats.skew(series)
    kurt = stats.kurtosis(series)
    jb_stat, jb_p = stats.jarque_bera(series)

    # --- Stationarity (ADF test) ---
    try:
        adf_result = adfuller(series, autolag="AIC")
        # adf_stat =  adf_result[0]
        adf_p = adf_result[1]
    except Exception as e:
        print(f"⚠️ ADF test failed: {e}")

    # --- Autocorrelation (Ljung-Box) ---
    lb_results = acorr_ljungbox(df, lags=[12], return_df=True)
    lbq_p = lb_results["lb_pvalue"].values[0]

    # # --- Ljung-Box on squared series (heteroskedasticity) ---
    lb2_results = acorr_ljungbox(df**2, lags=[lags], return_df=False)
    # lbq2_stat = lb2_results['lb_stat'].values[0]
    lbq2_p = lb2_results["lb_pvalue"].values[0]

    # --- ARCH LM test ---
    arch_results = het_arch(series)
    # arch_stat = arch_results[0]
    arch_p = arch_results[1]

    # --- Summary Table ---
    df_diag = pd.DataFrame(
        {
            "Statistic": [
                "Number of observations",
                "Mean",
                "Standard deviation",
                "Skewness",
                "Kurtosis",
                "Jarque-Bera (p)",
                "ADF (p)",
                "Ljung-Box Q-test (p)",
                "Ljung-Box Q²-test (p)",
                "ARCH test (p)",
            ],
            "Value": [
                n,
                round(mean, 4),
                round(std, 4),
                round(skew, 4),
                round(kurt, 4),
                round(jb_p, 4),
                round(adf_p, 4),
                round(lbq_p, 4),
                round(lbq2_p, 4),
                round(arch_p, 4),
            ],
        }
    )

    return df_diag


def plot_acf_pacf(df, lags=24, use_returns=False, title="ACF/PACF Diagnostics"):
    """
    Plot ACF and PACF for BACEN series (levels or returns),
    and return both the figure and diagnostic metrics.
    """

    # --- Prepare series ---
    if "Value" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "Value"})

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    if use_returns:
        series = df["Value"].pct_change() * 100
        series = series.dropna()
        label = "Returns – Percentage (Δy/yₜ₋₁ * 100)"
    else:
        series = df["Value"]
        label = "Levels – Raw Series"

    # --- Compute ACF/PACF ---
    acf_vals = acf(series, nlags=lags, fft=False)
    pacf_vals = pacf(series, nlags=lags, method="ywm")

    # --- Statistical significance threshold ---
    n = len(series)
    conf = 2 / np.sqrt(n)  # approx 95% confidence band
    sig_lags_acf = [i for i, v in enumerate(acf_vals) if abs(v) > conf]
    sig_lags_pacf = [i for i, v in enumerate(pacf_vals) if abs(v) > conf]

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    fig.suptitle(f"{title} – {label}", fontsize=12, fontweight="bold")

    plot_acf(series, ax=axes[0], lags=lags, alpha=0.05)
    axes[0].set_title("ACF")

    plot_pacf(series, ax=axes[1], lags=lags, alpha=0.05, method="ywm")
    axes[1].set_title("PACF")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # --- Diagnostic summary ---
    metrics = {
        "label": label,
        "n_obs": n,
        "acf_vals": acf_vals.tolist(),
        "pacf_vals": pacf_vals.tolist(),
        "significant_acf_lags": sig_lags_acf,
        "significant_pacf_lags": sig_lags_pacf,
    }

    return fig, metrics


def bacen_agent_final(
    prompt,
    dictionary=BACEN_TOP_SERIES,
    start=None,
    end=pd.Timestamp.now().normalize(),
    plot=True,
):
    df, code = bacen_agent_load(prompt, dictionary, start, end)
    if df is None or df.empty:
        return None

    if plot:
        bacen_agent_plot_full(df, code, prompt)

    return df


def infer_date_frequency_forecast(df):
    """
    Try to infer the frequency of a time series DataFrame with a Date index.
    Returns a pandas frequency string (e.g. 'W', 'M', 'Q', 'D') or None.
    """
    try:
        freq = pd.infer_freq(df.index)
        if freq is None:
            # fallback if pandas can’t infer directly - Create two objects ands subtrqact then pe rposition 
            diff = (df.index[1:] - df.index[:-1]).days
            # Now we calculate the average dates spacing 
            avg_diff = diff.mean()

            if avg_diff < 2:
                freq = "D"
            elif avg_diff < 10:
                freq = "W"
            elif avg_diff < 40:
                freq = "M"
            elif avg_diff < 120:
                freq = "Q"
            else:
                freq = "A"
        return freq
    except Exception:
        return "D"
    

def plot_residual_diagnostics(residuals):
    """Generate residual diagnostics: time plot, ACF, histogram, and QQ plot."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    #fig.suptitle(f"Residual Diagnostics", fontsize=10, fontweight="bold")

    # Residuals over time
    axes[0, 0].plot(residuals, color="black", linewidth=1, marker="o", markersize=3)
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Residuals vs Time")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Residual")

    # ACF Plot
    plot_acf(residuals, ax=axes[0, 1], lags=20, title="Autocorrelation (ACF)")

    # Histogram + KDE
    sns.histplot(residuals, kde=True, ax=axes[1, 0], color="gray")
    axes[1, 0].set_title("Residual Distribution")
    axes[1, 0].set_xlabel("Residuals")

    # QQ Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("QQ Plot (Normality Check)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def baseline_forecast(
    df,
    model_type="single",
    alpha=None,
    beta=None,
    gamma=None,
    season_periods=None,
    test_size=0.2,
    steps_ahead=30
):
    """
    Expanding-window baseline forecast (Single, Double, or Holt-Winters).

    - Starts with 80% of data and expands
    - Forecasts 30 steps ahead each iteration
    - Ends with full-sample out-of-sample 30-step forecast
    """

    # --- Data prep ---
    df = df.copy()
    if "Value" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "Value"})

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    y = df["Value"].dropna()

    # --- Split train/test ---
    split_idx = int(len(y) * (1 - test_size))
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # --- Default parameters ---
    rolling_results = []

    print(f"Expanding-window forecast: {steps_ahead} steps ahead per iteration")

    # --- Expanding window (walk-forward 30 steps at a time) ---
    for end_idx in range(split_idx, len(y) - steps_ahead + 1, steps_ahead):
        y_window = y.iloc[:end_idx]

        # Fit model based on type
        if model_type == "single":
            model = SimpleExpSmoothing(y_window)
            fit = model.fit(smoothing_level=alpha, optimized=(alpha is None))

        elif model_type == "double":
            model = ExponentialSmoothing(y_window, trend="add", seasonal=None)
            fit = model.fit(
                smoothing_level=alpha,
                smoothing_trend=beta,
                optimized=(alpha is None or beta is None)
            )

        elif model_type == "holtwinters":
            if season_periods is None:
                raise ValueError("season_periods must be provided for Holt-Winters model.")
            model = ExponentialSmoothing(
                y_window, trend="add", seasonal="add", seasonal_periods=season_periods
            )
            fit = model.fit(
                smoothing_level=alpha,
                smoothing_trend=beta,
                smoothing_seasonal=gamma,
                optimized=(alpha is None or beta is None or gamma is None)
            )
        else:
            raise ValueError("Invalid model_type")

        forecast_values = fit.forecast(steps_ahead)
        test_segment = y.iloc[end_idx:end_idx + steps_ahead]

        temp = pd.DataFrame({
            "Date": test_segment.index,
            "Actual": test_segment.values,
            "Forecast": forecast_values.values[:len(test_segment)],
            "type": "rolling_eval"
        })
        rolling_results.append(temp)

    # Combine rolling forecasts
    rolling_df = pd.concat(rolling_results, ignore_index=True) if rolling_results else pd.DataFrame()

    # --- Final full-sample model (out-of-sample forecast) ---
    if model_type == "single":
        final_model = SimpleExpSmoothing(y).fit(smoothing_level=alpha, optimized=(alpha is None))
    elif model_type == "double":
        final_model = ExponentialSmoothing(y, trend="add").fit(
            smoothing_level=alpha, smoothing_trend=beta, optimized=(alpha is None or beta is None)
        )
    else:
        final_model = ExponentialSmoothing(
            y, trend="add", seasonal="add", seasonal_periods=season_periods
        ).fit(
            smoothing_level=alpha,
            smoothing_trend=beta,
            smoothing_seasonal=gamma,
            optimized=(alpha is None or beta is None or gamma is None)
        )

    forecast_out = final_model.forecast(steps_ahead)
    last_date = y.index.max()
    freq = pd.infer_freq(y.index) or "D"
    future_dates = pd.date_range(last_date, periods=steps_ahead + 1, freq=freq)[1:]

    out_df = pd.DataFrame({
        "Date": future_dates,
        "Actual": np.nan,
        "Forecast": forecast_out.values,
        "type": "out_of_sample"
    })

    # --- Combine all forecasts ---
    forecast_df = pd.concat([rolling_df, out_df], ignore_index=True)

    # --- Metrics (on rolling only) ---
    mae = np.mean(np.abs(rolling_df["Actual"] - rolling_df["Forecast"]))
    mask = rolling_df["Actual"] != 0
    mape = np.mean(np.abs((rolling_df["Actual"][mask] - rolling_df["Forecast"][mask]) /
                            rolling_df["Actual"][mask])) * 100

    print(f"Baseline Expanding-window | MAPE: {mape:.2f}% | MAE: {mae:.2f}")

    # --- Residual diagnostics ---
    fitted = final_model.fittedvalues
    residuals = y - fitted
    resid_summary = {
        "mean": np.mean(residuals),
        "std": np.std(residuals),
        "skew": stats.skew(residuals),
        "kurtosis": stats.kurtosis(residuals)
    }

    # --- Final result dict ---
    result = {
        "forecast_df": forecast_df,
        "metrics": {"MAE": mae,
                     "MAPE": mape
        },
        "freq": freq,
        "df_processed": df,
        "residuals" : residuals,
        "residuals_summary": resid_summary,
        "model": final_model
    }

    return result



def prophet_forecast_standard(
        df,
        test_size=0.1,
        changepoint_prior_scale=0.5,
        changepoint_range=0.95,
        n_changepoints=500,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="additive",
        steps_ahead=30,
        interval_width=0.9
    ):
    """
    Prophet Forecast - Standard (Train/Test + Out-of-Sample)

    - Fits Prophet on 80% of the data (train)
    - Predicts all test dates (in-sample evaluation)
    - Forecasts next N future business days (out-of-sample)
    """

    # --- Data preparation ---
    df = df.copy()
    if "Value" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "Value"})

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    df_prophet = df.rename(columns={"Date": "ds", "Value": "y"})

    # --- Train/Test split ---
    split_idx = int(len(df_prophet) * (1 - test_size))
    train_df, test_df = df_prophet.iloc[:split_idx], df_prophet.iloc[split_idx:]

    # --- Model definition ---
    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        changepoint_range=changepoint_range,
        n_changepoints=n_changepoints,
        interval_width=interval_width,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        seasonality_mode=seasonality_mode.lower()
    )

    # Add more flexible short-term patterns
    model.add_seasonality(name='monthly', period=30.5, fourier_order=10)
    model.add_seasonality(name='quarterly', period=90, fourier_order=5)

    # --- Fit model ---
    print("🚀 Training Prophet model on training data...")
    model.fit(train_df)

    # --- Predict on test data (in-sample validation) ---
    forecast_test = model.predict(test_df[["ds"]])
    y_true = test_df["y"].values
    y_pred = forecast_test["yhat"].values

    # --- Metrics ---
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"📊 MAPE: {mape:.2f}% | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

    # --- Frequency detection ---
    freq = infer_date_frequency_forecast(df)

    if freq is None or freq == "D":
        # If weekends are missing, switch to business days
        if not df_prophet["ds"].dt.dayofweek.isin([5, 6]).any():
            freq = "B"
        else:
            freq = "D"

    # --- Future forecast (out-of-sample) ---
    print(f"📅 Detected frequency: {freq}")

    last_date = df_prophet["ds"].max()
    print(last_date)
    if freq == "B":
        # skip weekends
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps_ahead)
    else:
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps_ahead, freq=freq)

    future_df = pd.DataFrame({"ds": future_dates})
    forecast_out = model.predict(future_df)[["ds", "yhat"]]
    forecast_out = forecast_out.rename(columns={"ds": "Date", "yhat": "Forecast"})

    # --- Combine full forecast for plotting (train + test + future) ---
    forecast_test_series = pd.Series(y_pred, index=test_df["ds"], name="Forecast (Test)")
    forecast_full = pd.concat([train_df.set_index("ds")["y"], forecast_test_series], axis=0)

    # -- Residual Error Diagnostics --#
    residuals = y_true - y_pred
    resid_mean = np.mean(residuals)
    resid_std = np.std(residuals)

    # --- Return results ---
    result = {
        "train_df": train_df,
        "test_df": test_df,
        "test_forecast": forecast_test_series,
        "forecast_out": forecast_out,
        "forecast_full": forecast_full,
        "mape": mape,
        "mae": mae,
        "rmse": rmse,
        "freq": freq,
        "df_processed": df,
        "model": model,
        "residuals": residuals,
        "residuals_summary": {
            "mean": resid_mean,
            "std": resid_std,
            "skew": stats.skew(residuals),
            "kurtosis": stats.kurtosis(residuals)
        },
    }

    return result


def prophet_forecast_rolling(
        df,
        test_size=0.05,
        daily_seasonality=None,
        weekly_seasonality=None,
        yearly_seasonality=None,
        steps_ahead=1,
        show_progress=True,
        changepoint_prior_scale=0.05):
    """
    Rolling-window Prophet forecast with correct alignment, metrics, and stability.
    Similar to ARIMA one-step ahead rolling forecast 
    """

    # --- Data preparation ---
    df = df.copy()
    if "Value" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "Value"})

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Detect frequency dynamically
    freq = infer_date_frequency_forecast(df)
    print(f"📈 Detected frequency: {freq}")

    # Prophet expects 'ds' and 'y'
    df_prophet = df.rename(columns={"Date": "ds", "Value": "y"})

    split_idx = int(len(df_prophet) * (1 - test_size))
    train_df, test_df = df_prophet.iloc[:split_idx], df_prophet.iloc[split_idx:]

    preds = []
    test_dates = []

    n_fits = len(test_df)
    print(f"🔁 Running rolling Prophet ({n_fits} steps, {steps_ahead}-ahead horizon)")

    start_time = time.time()

    # --- Rolling forecast ---
    for i in tqdm(range(n_fits), desc="Rolling Forecast Progress", disable=not show_progress):
        train_window = pd.concat([train_df, test_df.iloc[:i]])

        # Include only non-None seasonalities
        model_params = {
            "changepoint_prior_scale": changepoint_prior_scale,
        }
        if daily_seasonality is not None:
            model_params["daily_seasonality"] = daily_seasonality
        if weekly_seasonality is not None:
            model_params["weekly_seasonality"] = weekly_seasonality
        if yearly_seasonality is not None:
            model_params["yearly_seasonality"] = yearly_seasonality

        model = Prophet(**model_params)
        model.fit(train_window)

        # --- Predict next step (aligned with test_df.iloc[i]) ---
        next_date = test_df.iloc[i]["ds"]
        future = pd.DataFrame({"ds": [next_date]})
        forecast = model.predict(future)

        preds.append(forecast["yhat"].values[0])
        test_dates.append(next_date)

        # Runtime estimate
        if i > 0 and i % 20 == 0:
            elapsed = time.time() - start_time
            avg_fit = elapsed / (i + 1)
            remaining = (n_fits - (i + 1)) * avg_fit / 60
            print(f"⏳ ~{remaining:.1f} min remaining...")

    rolling_forecast = pd.Series(preds, index=test_dates, name="Forecast")

    total_time = (time.time() - start_time) / 60
    print(f"✅ Rolling forecast finished in {total_time:.1f} minutes.")

    # --- Final refit on full data ---
    final_model = Prophet(**model_params)
    final_model.fit(df_prophet)
    future_full = final_model.make_future_dataframe(periods=steps_ahead, freq=freq)
    forecast_out = final_model.predict(future_full).tail(steps_ahead)[["ds", "yhat"]]
    forecast_out.columns = ["Date", "Forecast"]

    # --- Metrics ---
    y_true = test_df["y"].iloc[:len(rolling_forecast)]
    y_pred = rolling_forecast.values

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    print(f"📊 MAPE: {mape:.2f}% | RMSE: {rmse:.2f} | MAE: {mae:.2f}")

    # --- Output dictionary ---
    result = {
        "rolling_forecast": rolling_forecast,
        "forecast_out": forecast_out,
        "mape": mape,
        "mae": mae,
        "rmse": rmse,
        "freq": freq,
        "df_processed": df,
        "runtime_min": total_time,
    }

    return result


def prophet_forecast_standard_expanding_window(
        df,
        test_size=0.2,
        steps_ahead=30,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=0.8,
        n_changepoints=300,
        changepoint_range=1.0,
        show_progress=True
    ):
    """
    Prophet Expanding Window Forecast:
    1. Train on 80% of data.
    2. Forecast 'steps_ahead' steps ahead (on next test chunk).
    3. Expand training set with those predicted days.
    4. Repeat until reaching the end of data.
    5. Generate out-of-sample business-day forecast.
    """

    # --- Data preparation ---
    df = df.copy()
    if "Value" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "Value"})

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date")

    # --- Frequency detection (business-day aware) ---
    freq = infer_date_frequency_forecast(df)
    if freq is None or freq == "D":
        if not df["Date"].dt.dayofweek.isin([5, 6]).any():
            freq = "B"  # Business days only
        else:
            freq = "D"
    print(f"Detected frequency: {freq}")

    # --- Prophet expects columns ds/y ---
    df_prophet = df.rename(columns={"Date": "ds", "Value": "y"})
    df_prophet["y"] = np.log(df_prophet["y"]).rolling(5, center=True).mean()
    df_prophet = df_prophet.dropna(subset=["y"]).reset_index(drop=True)

    # --- Initial split ---
    split_idx = int(len(df_prophet) * (1 - test_size))
    full_train = df_prophet.iloc[:split_idx]
    full_test = df_prophet.iloc[split_idx:].reset_index(drop=True)

    preds, pred_dates = [], []

    total_test = len(full_test)
    n_iterations = int(np.ceil(total_test / steps_ahead))

    print(f"🔁 Running expanding forecast: {n_iterations} iterations | horizon={steps_ahead}")

    start_time = time.time()

    # --- Main expanding-window loop ---
    for i in tqdm(range(n_iterations), disable=not show_progress):
        start_idx = i * steps_ahead
        end_idx = min((i + 1) * steps_ahead, total_test)

        test_chunk = full_test.iloc[start_idx:end_idx]
        if len(test_chunk) == 0:
            break

        print(f"🧩 Iter {i+1}/{n_iterations} | Train size: {len(full_train)} | Forecast {len(test_chunk)} steps ahead")

        # --- Train Prophet ---
        model = Prophet(
            seasonality_mode=seasonality_mode,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
            n_changepoints=n_changepoints,
            changepoint_range=changepoint_range
        )
        model.add_seasonality('monthly', period=30.5, fourier_order=10)
        model.add_seasonality('quarterly', period=90, fourier_order=5)
        model.fit(full_train)

        # --- Predict current chunk ---
        forecast = model.predict(test_chunk[["ds"]])
        preds.extend(np.exp(forecast["yhat"].values))
        pred_dates.extend(test_chunk["ds"].values)

        # --- Expand training window ---
        full_train = pd.concat([full_train, test_chunk], ignore_index=True)

    # --- Combine results ---
    rolling_forecast = pd.Series(preds, index=pd.to_datetime(pred_dates), name="Rolling Forecast")

    # Ensure full_train contains the complete dataset before final forecast
    if len(full_train) < len(df_prophet):
        remaining = df_prophet.iloc[len(full_train):]
        if not remaining.empty:
            print(f"🧩 Appending {len(remaining)} remaining unseen rows to full_train.")
            full_train = pd.concat([full_train, remaining], ignore_index=True)

    # --- Evaluate metrics ---
    actual = df_prophet.set_index("ds").loc[rolling_forecast.index, "y"]
    actual = np.exp(actual)
    mape = np.mean(np.abs((actual - rolling_forecast) / actual)) * 100
    mae = np.mean(np.abs(actual - rolling_forecast)) 
    rmse = np.sqrt(np.mean((actual - rolling_forecast) ** 2))

    print(f"📊 Final Rolling MAPE: {mape:.2f}% | RMSE: {rmse:.2f} | ⏱ {time.time()-start_time:.1f}s")

    # -- Error diagnostics
    residuals = actual - rolling_forecast
    resid_mean = np.mean(residuals)
    resid_std = np.std(residuals)
 
    # --- Final model (fit on all data) ---
    final_model = Prophet(
        seasonality_mode=seasonality_mode,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        n_changepoints=n_changepoints,
        changepoint_range=changepoint_range
    )
    final_model.add_seasonality('monthly', period=30.5, fourier_order=10)
    final_model.add_seasonality('quarterly', period=90, fourier_order=5)
    final_model.fit(df_prophet)

    # --- Build future (out-of-sample) forecast ---
    last_date = df_prophet["ds"].max()
    print(last_date)
    if freq == "B":
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps_ahead)
    else:
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps_ahead, freq=freq)

    future_df = pd.DataFrame({"ds": future_dates})
    forecast_out = final_model.predict(future_df)[["ds", "yhat"]]
    forecast_out.columns = ["Date", "Forecast"]
    forecast_out["Forecast"] = np.exp(forecast_out["Forecast"])

    # --- Return results ---
    results = {
        "rolling_forecast": rolling_forecast,
        "mape": mape,
        "rmse": rmse,
        "mae": mae,
        "forecast_out": forecast_out,
        "freq": freq,
        "df_processed": df,
        "model": final_model,
        "residuals": residuals,
        "residuals_summary": {
            "mean": resid_mean,
            "std": resid_std,
            "skew": stats.skew(residuals),
            "kurtosis": stats.kurtosis(residuals)
        },
    }

    return results


def prophet_forecast_standard_rolling_mean(
        df,
        test_size=0.1,
        weekly_seasonality=True,
        yearly_seasonality=True,
        steps_ahead=30,
        seasonality_mode="additive",
        changepoint_prior_scale = 0.8,  # or even 1.0
        n_changepoints = 300,
        changepoint_range = 1.0
    ):
    """
    Prophet Forecast with log smoothing and short-term seasonalities.
    """

    # --- Data preparation ---
    df = df.copy()
    if "Value" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "Value"})

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Prophet expects 'ds' and 'y'
    df_prophet = df.rename(columns={"Date": "ds", "Value": "y"})
    df_prophet["y"] = np.log(df_prophet["y"])                # log transform
    df_prophet["y"] = df_prophet["y"].rolling(5, center=True).mean()  # smooth
    df_prophet = df_prophet.dropna(subset=["y"])             # remove NaNs

    # --- Split train/test ---
    split_idx = int(len(df_prophet) * (1 - test_size))
    train_df, test_df = df_prophet.iloc[:split_idx], df_prophet.iloc[split_idx:]

    # --- Prophet model ---
    model = Prophet(
        seasonality_mode=seasonality_mode,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        changepoint_prior_scale = 0.8,  # or even 1.0
        n_changepoints = 300,
        changepoint_range = 1.0

    )
    model.add_seasonality('monthly', period=30.5, fourier_order=10)
    model.add_seasonality('quarterly', period=90, fourier_order=5)

    # --- Fit ---
    print("🚀 Training Prophet model...")
    model.fit(train_df)

    # --- Predict on test period ---
    forecast_test = model.predict(test_df[["ds"]])
    y_true = np.exp(test_df["y"].values)
    y_pred = np.exp(forecast_test["yhat"].values)

    # --- Metrics ---
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"📊 MAPE: {mape:.2f}% | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

    # --- Future forecast ---
    freq = pd.infer_freq(df_prophet["ds"]) or "D"
    future = model.make_future_dataframe(periods=steps_ahead, freq=freq)
    forecast_full = model.predict(future)

    forecast_out = (
        forecast_full.tail(steps_ahead)[["ds", "yhat"]]
        .rename(columns={"ds": "Date", "yhat": "Forecast"})
    )
    forecast_out["Forecast"] = np.exp(forecast_out["Forecast"])

    # --- Return ---
    result = {
        "train_df": train_df,
        "test_df": test_df,
        "test_forecast": pd.Series(y_pred, index=test_df["ds"], name="Forecast (Test)"),
        "forecast_out": forecast_out,
        "mape": mape,
        "mae": mae,
        "rmse": rmse,
        "freq": freq,
        "df_processed": df,
        "model": model
    }

    return result


'''initialize CHRONOS pipeline'''
pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-tiny")

def aws_chromos(
    df,
    pipeline,
    test_size=0.2,
    prediction_length=30,
    steps_ahead=30,
    num_samples=20,
):
    """
    AWS Chronos re-feed strategy (Streamlit-compatible)
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'timestamp' (or 'Date') and one numeric target column.
    pipeline : Chronos pipeline
        Pre-trained Chronos model (already loaded via `ChronosPipeline.from_pretrained()`).
    test_size : float
        Fraction of data to use for test split.
    prediction_length : int
        Forecast horizon.
    steps_ahead : int
        Step size for rolling window evaluation.
    num_samples : int
        Number of forecast samples to draw.
    """

    # --- Data preparation ---
    df = df.copy()
    if "target" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "target"})
    if "timestamp" not in df.columns:
        df = df.rename(columns={df.columns[0]: "timestamp"})

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)


    # --- Rolling forecast parameters ---
    initial_train_size = int((1 - test_size) * len(df))
    results = []

    print('Starting Re-feeding different historical slices')
    # --- Rolling in-sample forecasts ---
    for end_idx in tqdm(range(initial_train_size, len(df) - prediction_length, steps_ahead)):
        train_df = df.iloc[:end_idx]
        test_df = df.iloc[end_idx:end_idx + prediction_length]

        # Convert to tensor
        values = torch.tensor(train_df["target"].values, dtype=torch.float32).unsqueeze(0)

        # Forecast using pre-trained Chronos
        preds = pipeline.predict(values, prediction_length=prediction_length, num_samples=num_samples)
        forecast_np = preds.mean(dim=1).detach().cpu().numpy().flatten()

        # Store results
        temp = pd.DataFrame({
            "timestamp": test_df["timestamp"].values,
            "true_value": test_df["target"].values,
            "forecast_value": forecast_np,
            "type": "rolling_eval"
        })
        results.append(temp)

    # Combine all rolling forecasts
    results_df = pd.concat(results, ignore_index=True)

    # -- Error diagnostics
    residuals = results_df['true_value'] - results_df['forecast_value']
    resid_mean = np.mean(residuals)
    resid_std = np.std(residuals)

    # --- Out-of-sample forecast ---
    values_out = torch.tensor(df["target"].values, dtype=torch.float32).unsqueeze(0)
    preds_out = pipeline.predict(values_out, prediction_length=prediction_length, num_samples=num_samples)
    forecast_np_out = preds_out.mean(dim=1).detach().cpu().numpy().flatten()

    # Future timestamps
    last_date = pd.to_datetime(df["timestamp"]).max()
    
    # --- Frequency detection (business-day aware) ---
    freq = infer_date_frequency_forecast(df)
    if freq is None or freq == "D":
        if not df["timestamp"].dt.dayofweek.isin([5, 6]).any():
            freq = "B"  # Business days only
        else:
            freq = "D"
    print(f"Detected frequency: {freq}")

    future_dates = pd.date_range(last_date, periods=prediction_length + 1, freq=freq)[1:]

    # Combine
    out_df = pd.DataFrame({
        "timestamp": future_dates,
        "true_value": np.nan,
        "forecast_value": forecast_np_out,
        "type": "out_of_sample"
    })

    # Append together
    results_df = pd.concat([results_df, out_df], ignore_index=True)

    # --- Compute metrics (only for in-sample part) ---
    eval_df = results_df[results_df["type"] == "rolling_eval"].dropna()

    mae = mean_absolute_error(eval_df["true_value"], eval_df["forecast_value"])

    mask = eval_df["true_value"] != 0
    mape = np.mean(np.abs(
        (eval_df["true_value"][mask] - eval_df["forecast_value"][mask]) /
        eval_df["true_value"][mask]
    )) * 100

    # --- Return dictionary ---
    results = {
        "df_processed": df,
        "results_df_aws": results_df,
        "metrics": {
            "MAE": mae,
            "MAPE": mape,
        },
        "freq": freq,
        "model": pipeline,
        "residuals": residuals,
        "residuals_summary": {
            "mean": resid_mean,
            "std": resid_std,
            "skew": stats.skew(residuals),
            "kurtosis": stats.kurtosis(residuals)
        },
    }

    return results


def run_arima(
    df,
    test_size: float = 0.2,
    p_lags: int | None = None,
    q_lags: int | None = None,
    d: int = 0,
    steps_ahead: int = 30
):
    """
    Expanding-window ARIMA(p,d,q) forecast.
    Starts with 80% of data, predicts next 30 steps repeatedly,
    then performs a final out-of-sample 30-step forecast.

    Returns
    -------
    dict with:
        - forecast_df: rolling + out-of-sample forecasts
        - metrics: MAE, RMSE, MAPE
        - model: last fitted ARIMA model
        - freq: detected frequency
    """

    # --- Data preparation ---
    df = df.copy()
    if "Value" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "Value"})
    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["Value"])

    # --- Rolling forecast parameters ---
    initial_train_size = int((1 - test_size) * len(df))
    p = p_lags if p_lags is not None else 1
    q = q_lags if q_lags is not None else 1
    results = []

    # --- Expanding-window evaluation ---
    print('Starting ARIMA Expanding-window evaluation')
    for end_idx in range(initial_train_size, len(df) - steps_ahead, steps_ahead):
        train_df = df.iloc[:end_idx]
        test_df = df.iloc[end_idx:end_idx + steps_ahead]

        model = ARIMA(train_df["Value"], order=(p, d, q))
        fitted = model.fit()
        forecast = fitted.forecast(steps=steps_ahead)
        temp = pd.DataFrame({
            "Date": test_df["Date"].values,
            "actual": test_df["Value"].values,
            "Forecast": forecast.values,
            "type": "rolling_eval"
        })
        results.append(temp)


    rolling_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    # -- Error diagnostics
    residuals = rolling_df['actual'] - rolling_df['Forecast']
    resid_mean = np.mean(residuals)
    resid_std = np.std(residuals)


    # --- Final out-of-sample forecast ---
    full_model = ARIMA(df["Value"], order=(p, d, q))
    full_fit = full_model.fit()
    forecast_out = full_fit.forecast(steps=steps_ahead)

    last_date = df["Date"].iloc[-1]
    freq = pd.infer_freq(df["Date"]) or "D"
    future_dates = pd.date_range(last_date, periods=steps_ahead + 1, freq=freq)[1:]

    out_df = pd.DataFrame({
        "Date": future_dates,
        "actual": np.nan,
        "Forecast": forecast_out.values,
        "type": "out_of_sample"
    })

    # --- Merge rolling + out-of-sample results ---
    forecast_df = pd.concat([rolling_df, out_df], ignore_index=True)

    # --- Compute metrics (only for in-sample part) ---

    mae = mean_absolute_error(rolling_df["actual"], rolling_df["Forecast"])
    mask = rolling_df["actual"] != 0
    mape = np.mean(np.abs((rolling_df["actual"][mask] - rolling_df["Forecast"][mask]) /
                            rolling_df["actual"][mask])) * 100


    # --- Return everything ---
    results = {
        "df_processed": df,
        "forecast_df": forecast_df,
        "metrics": {
            "MAE": mae,
            "MAPE": mape
        },
        "model": full_fit,
        "residuals": residuals,
        "residuals_summary": {
            "mean": resid_mean,
            "std": resid_std,
            "skew": stats.skew(residuals),
            "kurtosis": stats.kurtosis(residuals)
        },
    }

    return results

