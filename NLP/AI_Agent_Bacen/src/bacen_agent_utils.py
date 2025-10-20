# BACEN AI Agent -  source functions

# Similarity
import difflib
import re
import unicodedata
import warnings
from difflib import SequenceMatcher

import matplotlib.pyplot as plt
import pandas as pd  # lib for data manipulation
import requests  # access HTTP content
from bcb import sgs
from requests.adapters import HTTPAdapter, Retry
## Timne-series diagnostics tests
from scipy import stats
## langChain
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from tqdm import trange  # lib for progress bars

warnings.filterwarnings("ignore")


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
    "pib produto interno bruto trimestral": 4380,
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


def bacen_agent_load_langchain(
    prompt,
    dictionary=BACEN_TOP_SERIES,
    start=None,
    end=None,
    cutoff=0.35,
    embedder=None,
):
    """
    Intelligent BACEN data retriever & analyzer using sentence_transformers embeddings.
    1. Uses semantic similarity (SentenceTransformer) to match prompt → BACEN code.
    2. Fetches data automatically.
    3. Returns df, code, and similarity score.
    """

    if start is None:
        start = "2020-01-01"

    if end is None:
        end = pd.Timestamp.now().normalize()

    if embedder is None:
        raise ValueError(
            "Embedder not provided. Please pass a SentenceTransformer instance."
        )

    # --- Step 1: Vectorize your BACEN dictionary ( split keys and values) ---
    bacen_keys = list(dictionary.keys())
    bacen_codes = list(dictionary.values())
    bacen_embs = embedder.encode(
        bacen_keys, convert_to_numpy=True, normalize_embeddings=True
    )

    # --- Step 2: Semantic search ---
    q_emb = embedder.encode([prompt], normalize_embeddings=True)
    sims = cosine_similarity(q_emb, bacen_embs)[0]
    top_idx = sims.argmax()

    best_key = bacen_keys[top_idx]
    best_code = bacen_codes[top_idx]
    best_sim = sims[top_idx]

    print(
        f"Best series selected: '{best_key}' | Code: {best_code} | Similarity: {best_sim:.2f}"
    )

    # --- Step 3: Fetch BACEN data ---
    try:
        df = sgs.get(best_code, start=start, end=end)
        df = df.reset_index().rename(columns={"index": "Date", best_code: "Value"})
        print(f"Retrieved {len(df)} rows for code {best_code}.")
    except Exception as e:
        print(f"❌ Error retrieving series {best_code}: {e}")
        return pd.DataFrame(), best_code

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
    """Plot ACF and PACF for BACEN series (levels and/or returns)."""
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    fig.suptitle(f"{title} – {label}", fontsize=12, fontweight="bold")

    plot_acf(series, ax=axes[0], lags=lags, alpha=0.05)
    axes[0].set_title("ACF")

    plot_pacf(series, ax=axes[1], lags=lags, alpha=0.05, method="ywm")
    axes[1].set_title("PACF")

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


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
