import warnings, unicodedata
from collections import Counter
import nltk
from loguru import logger

warnings.filterwarnings("ignore")

# =========================
# BLOCO NLTK (MANTER)
# =========================
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# ------- utilidades -------
NORMALIZAR = True  # coloque False se quiser comparar “cru” (sem remover acentos)

def normalize(s: str) -> str:
    if not NORMALIZAR:
        return s.casefold()
    s = s.casefold()
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn") # Remover acentos

def find_indices(tokens, target):
    return [i for i, t in enumerate(tokens) if t == target]

def context_snippets(tokens_raw, idxs, window=5, max_examples=3):
    ex = []
    for i in idxs[:max_examples]:
        a = max(0, i - window)
        b = min(len(tokens_raw), i + window + 1)
        ex.append(" ".join(tokens_raw[a:b]))
    return ex

# ------- leitura -------
txt_file = "transcricao.txt"
logger.info("Lendo arquivo {}...", txt_file)
with open(txt_file, "r", encoding="utf-8") as f:
    text = f.read()

# ------- tokenização (NLTK) -------
logger.info("Tokenizando texto (pt-BR) com NLTK...")
tokens_raw = nltk.word_tokenize(text, language="portuguese")  # <-- mantém NLTK
tokens = [normalize(t) for t in tokens_raw]
tok_set = set(tokens)
tok_counter = Counter(tokens)

# ------- busca -------
keywords = ["clabim", "santander", " vale"]
kw_norm = [normalize(w) for w in keywords]

results = {}
for original, kw in zip(keywords, kw_norm):
    idxs = find_indices(tokens, kw)
    results[original] = {
        "found": kw in tok_set,
        "count": tok_counter.get(kw, 0),
        "contexts": context_snippets(tokens_raw, idxs, window=5, max_examples=3),
    }

# ------- logs -------
for k, info in results.items():
    if info["found"]:
        logger.info("✅ '{}' encontrada {}x.", k, info["count"])
        for c in info["contexts"]:
            logger.info("   • contexto: {}", c)
    else:
        logger.info("❌ '{}' NÃO foi encontrada.", k)

logger.info("Resumo: {}", results)
