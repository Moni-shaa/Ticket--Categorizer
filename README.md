# Auto Email / Ticket Categorizer

Classifies incoming support tickets (subject + body) into one of four
categories — **Billing, Technical, HR, General** — using TF-IDF features and
a Logistic Regression classifier.

## Files
- `tickets_dataset.csv` — 40 hand-written labeled tickets (10 per category) used for training/testing.
- `ticket_categorizer.py` — full pipeline: preprocessing → TF-IDF → training → evaluation → predictions on 5 new sample tickets → bonus features.

## How to run

```bash
python3 ticket_categorizer.py
```

This will:
1. Load and clean the dataset
2. Vectorize the text with TF-IDF
3. Train a Logistic Regression classifier on a train/test split
4. Print accuracy, a full precision/recall/F1 report, and a confusion matrix
5. Predict the category (with confidence %) for 5 new, unseen sample tickets
6. Print a short reflection note

To also launch the interactive live demo (type a ticket, get an instant
prediction), run:

```bash
python3 ticket_categorizer.py --demo
```

## Pipeline

1. **Text preprocessing** — lowercasing, URL/punctuation/number removal,
   stopword filtering.
2. **Feature representation** — TF-IDF (unigrams), which down-weights words
   common to every ticket ("please", "thanks") and up-weights words
   distinctive to a category ("invoice", "server", "leave").
3. **Model** — `LogisticRegression`. Chosen over `MultinomialNB` here because
   with a small dataset it gives more reliable `predict_proba` scores, which
   the confidence-score bonus depends on. Swapping in `MultinomialNB()` is a
   one-line change in `train_model()` if you want to compare.
4. **Evaluation** — accuracy, per-class precision/recall/F1, and a confusion
   matrix, printed with a plain-English explanation of how to read it.

## Bonus features implemented
- **Confidence score** — every prediction returns a probability, not just a label.
- **Needs-human-review threshold** — predictions below 60% confidence are flagged for manual review instead of auto-assigned.
- **Priority tagging** — a simple keyword rule (e.g. "down", "urgent", "crash") tags tickets as `urgent` or `normal`.
- **Mini live demo** — `--demo` flag opens a CLI loop where you can type a ticket and see it classified instantly.
- **Reflection note** — printed at the end of the run, also below.

## Reflection
With more data, I'd want at least a few hundred examples per category (this
demo uses ~10 per class — enough to prove the pipeline works, but too small
to fully trust the accuracy number). More real historical tickets would also
expose messier language the clean synthetic set doesn't have. With more time
I'd add k-fold cross-validation instead of a single train/test split, compare
Naive Bayes vs. Logistic Regression side by side, and add a fallback
"unclear" label for tickets that genuinely straddle two categories.
