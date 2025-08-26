import streamlit as st
import nltk
from nltk.corpus import brown
from collections import Counter
from src.functions import get_candidates_v6

# one-time corpus download (no-op if already present)
nltk.download("brown", quiet=True)

@st.cache_resource
def load_vocab():
    return Counter(w.lower() for w in brown.words())

vocab = load_vocab()

def correct_word(word):
    cands = get_candidates_v6(word, vocab=vocab, top_n=3)
    best = cands[0]["candidate"] if cands else word
    return best, cands

def correct_sentence(sentence):
    words = sentence.split()
    corrected = []
    per_word = []
    for w in words:
        best, cands = correct_word(w)
        corrected.append(best)
        per_word.append((w, cands))   # keep original and its candidates
    return " ".join(corrected), per_word

st.title("Max Autocorrect App")
user_input = st.text_input("Type a sentence to correct:")

if st.button("Correct it!"):
    if user_input.strip():
        corrected, details = correct_sentence(user_input)
        st.success(f"✅ Corrected: {corrected}")

        st.subheader("Suggestions per word")
        for original, cands in details:
            if not cands:
                continue
            st.markdown(f"**{original}**")
            # cands is already a list of dicts with the right keys
            st.table([
                {
                    "candidate": c["candidate"],
                    "distance": c["distance"],
                    "freq": c["freq"],
                    "score": round(c["score"], 2)
                }
                for c in cands
            ])
    else:
        st.warning("Please enter some text.")
