"""Build data/vocab_1500.txt: the A2-or-easier lemma allow-list.

Three layers, all lemmatized through the SAME spaCy pipeline used at scoring
time (reward.nlp) and deduped, so list-membership and token lemmas stay
definitionally consistent ("went"->"go" can never silently miss the list):

  1. Oxford 3000-by-CEFR, A1 + A2 entries (~1.68k words) — the curated core
     of what a learner at this level is expected to know.
  2. A wordfreq top-5000 frequency backstop. The Oxford list is British and
     small; without this, common American spellings and everyday words
     (mom, favorite, neighbor, excited, building) get flagged as "hard".
     Measured: with the backstop, Opus A2 references scored *with source*
     hit mean 1.00 and the only residual flags are genuinely rare/domain
     words (ditch, pollen, slavic) — exactly what should be flagged.
  3. Closed-class words (pronouns, auxiliaries, determiners, prepositions,
     conjunctions, number words, particles) appended explicitly — curated
     lists handle these inconsistently and we never want "the"/"of" flagged.

The filename keeps the spec's `vocab_1500.txt` path; the actual list is
~5k lemmas after the backstop.

Run:  uv run python scripts/build_vocab_list.py
Then run the vocab term over the Opus A2 references with source
(scripts/validate_reward.py) — they should score ~1.0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wordfreq import top_n_list  # noqa: E402

from claro.reward.nlp import nlp, word_lemmas  # noqa: E402

# Frequency backstop: lemmatized top-N English words unioned onto the Oxford
# core. 5000 zeros out false-positives on the real (with-source) task while
# leaving genuinely rare words flagged.
WORDFREQ_TOPN = 5000


OXFORD_CEFR = ROOT / "data" / "raw" / "oxford_cefr.txt"
OUT = ROOT / "data" / "vocab_1500.txt"

# Closed-class lemmas appended after lemmatization. These are the function
# words an A2 reader treats as free; curated content-word lists handle them
# inconsistently. Lowercase lemmas only — they get unioned and deduped with
# the lemmatized Oxford entries.
CLOSED_CLASS = {
    # pronouns (personal, possessive, demonstrative, relative, indefinite)
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "mine", "yours", "hers", "ours", "theirs",
    "myself", "yourself", "himself", "herself", "itself",
    "ourselves", "yourselves", "themselves",
    "this", "that", "these", "those",
    "who", "whom", "whose", "which", "what",
    "one", "ones", "someone", "anyone", "everyone", "no_one",
    "something", "anything", "everything", "nothing",
    "somebody", "anybody", "everybody", "nobody",
    "somewhere", "anywhere", "everywhere", "nowhere",
    # be / have / do
    "be", "am", "is", "are", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "done", "doing",
    # modals
    "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "ought", "need", "dare", "used",
    # determiners / quantifiers
    "the", "a", "an", "some", "any", "no", "every", "each",
    "either", "neither", "all", "both", "half", "several",
    "enough", "much", "many", "more", "most", "few", "little",
    "less", "least", "another", "other", "such", "same",
    # prepositions
    "in", "on", "at", "to", "from", "of", "with", "without",
    "by", "for", "about", "as", "into", "onto", "off",
    "over", "under", "above", "below", "between", "among",
    "through", "during", "before", "after", "since", "until",
    "till", "against", "toward", "towards", "upon", "within",
    "behind", "beside", "beyond", "near", "out", "up", "down",
    "around", "along", "across", "past", "per", "via",
    "despite", "throughout", "underneath", "inside", "outside",
    # conjunctions / connectors
    "and", "or", "but", "nor", "so", "yet", "because",
    "although", "though", "while", "whereas", "if", "unless",
    "when", "whenever", "where", "wherever", "than", "whether",
    "once", "however", "therefore", "thus", "also",
    # number words
    "zero", "two", "three", "four", "five", "six",
    "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty",
    "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion", "trillion",
    "first", "second", "third", "fourth", "fifth", "sixth",
    "seventh", "eighth", "ninth", "tenth",
    # common particles / degree / polarity / discourse
    "not", "yes", "too", "very", "just", "only", "even",
    "still", "well", "here", "there", "now", "then", "again",
    "ever", "never", "always", "often", "sometimes", "usually",
    "please", "thank", "thanks", "ok", "okay", "hello", "hi",
    "yeah", "oh",
}


def build() -> list[str]:
    levels = json.loads(OXFORD_CEFR.read_text())
    raw_words = sorted({w.lower() for w in levels["A1"]} | {w.lower() for w in levels["A2"]})

    lemmas: set[str] = set()
    for word in raw_words:
        word_out = word_lemmas(word)
        lemmas.update(word_out)
        # Some entries lemmatize to nothing alphabetic (e.g. "CD"); keep the
        # bare lowercased surface so the original entry is never lost.
        if not word_out and word.strip().isalpha():
            lemmas.add(word.strip().lower())

    # Frequency backstop, lemmatized through the same pipeline (batched).
    freq_words = [w for w in top_n_list("en", WORDFREQ_TOPN) if w.isalpha()]
    for doc in nlp().pipe(freq_words, batch_size=256):
        for tok in doc:
            if tok.is_alpha:
                lemmas.add(tok.lemma_.lower())

    lemmas |= CLOSED_CLASS
    return sorted(lemmas)


def main() -> None:
    vocab = build()
    OUT.write_text("\n".join(vocab) + "\n")
    print(f"wrote {len(vocab)} lemmas -> {OUT}")
    print("first 20:", vocab[:20])
    print("last 20:", vocab[-20:])


if __name__ == "__main__":
    main()
