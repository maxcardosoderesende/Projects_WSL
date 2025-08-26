import textdistance



# --- CANDIDATE GENERATOR FUNCTION ---
def get_candidates_v5(word, vocab, max_dist=2, top_n=5):
    # Function to return the correct word candidates for a given word
    word = word.lower()
    candidates = []
    for w in vocab:
        if w[0] != word[0]:
            continue

        dist = textdistance.levenshtein(word, w)
        if dist <= max_dist:
            freq = vocab[w]
            prefix_bonus = 1.5 if w.startswith(word[:3]) else 0
            score = -4 * dist + 0.6 * (freq ** 0.5) + prefix_bonus
            candidates.append((w, dist, freq, score))
    sorted_candidates = sorted(candidates, key=lambda x: x[3], reverse=True)

    return [(w, d, f) for w, d, f, _ in sorted_candidates[:top_n]]


### --- CANDIDATE GENERATOR FUNCTION V6 ---
import textdistance

def get_candidates_v6(word, vocab, max_dist=2, top_n=3):
    word = word.lower()
    candidates = []

    for w in vocab:
        if not w:
            continue
        if w[0] != word[0]:           # simple prefix gate
            continue

        dist = textdistance.levenshtein(word, w)
        if dist <= max_dist:
            freq = vocab[w]
            prefix_bonus = 1.5 if w.startswith(word[:3]) else 0
            score = -4 * dist + 0.6 * (freq ** 0.5) + prefix_bonus
            candidates.append({
                "candidate": w,
                "distance": dist,
                "freq": freq,
                "score": score
            })

    # sort by score DESC and return top 3
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_n]

