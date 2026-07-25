"""
Auto Email / Ticket Categorizer
--------------------------------
Reads a labeled dataset of support tickets (subject + body), cleans the text,
converts it into TF-IDF features, trains a text classifier (Naive Bayes or
Logistic Regression), evaluates it, and predicts categories for new tickets.

Categories: Billing, Technical, HR, General

Bonus features implemented:
  1. Confidence score output (predict_proba)
  2. "Needs human review" threshold (confidence < 60%)
  3. Priority tagging (urgent/normal keyword rules)
  4. Mini live demo (CLI input loop)
  5. Reflection note (see bottom of this file / printed at the end)
"""

import re
import string
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

DATA_PATH = "tickets_dataset.csv"
RANDOM_STATE = 42
CONFIDENCE_THRESHOLD = 0.60  # below this -> needs human review

# A small, hand-picked stopword list so we don't need an extra NLTK download.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "to", "of",
    "in", "on", "at", "for", "with", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "it", "its", "i",
    "you", "your", "we", "our", "they", "their", "my", "me", "us", "as",
    "do", "does", "did", "have", "has", "had", "not", "will", "would",
    "can", "could", "should", "please", "just", "about", "from", "up",
    "out", "there", "here", "am", "im",
}

URGENT_KEYWORDS = {
    "down", "urgent", "not working", "asap", "immediately", "critical",
    "crash", "crashes", "broken", "emergency", "production", "outage",
}


# ---------------------------------------------------------------------------
# 1. TEXT PREPROCESSING
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/numbers/extra whitespace, remove stopwords."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # urls
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)                      # numbers
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def load_and_prepare_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    df["clean_text"] = df["text"].apply(clean_text)
    return df


# ---------------------------------------------------------------------------
# 2. FEATURE REPRESENTATION + 3. MODEL TRAINING
# ---------------------------------------------------------------------------
def train_model(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["category"],
        test_size=0.25, random_state=RANDOM_STATE, stratify=df["category"],
    )

    # TF-IDF turns text into weighted word-importance vectors: common words
    # across ALL tickets (like "please") get down-weighted, while words that
    # are distinctive to a category (like "invoice" or "server") get more
    # weight. That's why it beats raw word counts for this kind of task.
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Logistic Regression is used here (rather than Naive Bayes) because with
    # only a few dozen examples and bigram TF-IDF features, LogisticRegression
    # tends to be more stable and gives well-calibrated predict_proba scores,
    # which the confidence-score bonus feature depends on. Naive Bayes is a
    # perfectly valid alternative for this task (it's fast and works well on
    # small bag-of-words text data) - swap in MultinomialNB() below if you'd
    # like to compare.
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=10)
    model.fit(X_train_vec, y_train)

    return model, vectorizer, X_test_vec, y_test


# ---------------------------------------------------------------------------
# 4. EVALUATION
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.2%}\n")

    print("Classification report (precision / recall / f1 per category):")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print("Labels:", labels)
    print(cm)
    print(
        "\nHow to read it: each row is the TRUE category, each column is what "
        "the model PREDICTED. Diagonal values are correct predictions; any "
        "off-diagonal value is a specific type of mistake (e.g. row=Technical, "
        "col=General means a Technical ticket was mis-classified as General)."
    )


# ---------------------------------------------------------------------------
# BONUS: confidence score + human-review flag + priority tagging
# ---------------------------------------------------------------------------
def tag_priority(raw_text: str) -> str:
    lowered = raw_text.lower()
    for kw in URGENT_KEYWORDS:
        if kw in lowered:
            return "urgent"
    return "normal"


def predict_ticket(model, vectorizer, subject: str, body: str) -> dict:
    raw_text = f"{subject} {body}"
    cleaned = clean_text(raw_text)
    vec = vectorizer.transform([cleaned])

    probs = model.predict_proba(vec)[0]
    classes = model.classes_
    best_idx = probs.argmax()
    category = classes[best_idx]
    confidence = probs[best_idx]

    needs_review = confidence < CONFIDENCE_THRESHOLD
    priority = tag_priority(raw_text)

    return {
        "subject": subject,
        "predicted_category": category,
        "confidence": round(float(confidence), 3),
        "needs_human_review": needs_review,
        "priority": priority,
    }


# ---------------------------------------------------------------------------
# 5 new sample tickets (written by hand, not in the training set)
# ---------------------------------------------------------------------------
NEW_SAMPLE_TICKETS = [
    (
        "Server outage - production down",
        "Our production server has been completely unreachable for 20 minutes "
        "and customers cannot check out. This is urgent, please escalate immediately.",
    ),
    (
        "Overcharged on last invoice",
        "I noticed my last invoice includes a $30 charge I don't recognize. "
        "Could someone from billing review my account and issue a refund if it's an error?",
    ),
    (
        "Need to update emergency contact",
        "I'd like to update my emergency contact information on file with HR "
        "and also confirm my current leave balance for this year.",
    ),
    (
        "Just wanted to say thanks",
        "I don't have a specific issue, just wanted to pass along positive "
        "feedback about the support team's response time this week.",
    ),
    (
        "App keeps freezing on the reports page",
        "The reports page in the dashboard freezes every time I try to export "
        "a CSV. It worked fine last week but now it crashes the whole app.",
    ),
]


def run_predictions_on_samples(model, vectorizer):
    print("\n" + "=" * 70)
    print("PREDICTIONS ON 5 NEW UNSEEN SAMPLE TICKETS")
    print("=" * 70)
    for subject, body in NEW_SAMPLE_TICKETS:
        result = predict_ticket(model, vectorizer, subject, body)
        review_flag = " <-- NEEDS HUMAN REVIEW" if result["needs_human_review"] else ""
        print(
            f"\nSubject: {result['subject']}\n"
            f"  Predicted category : {result['predicted_category']}\n"
            f"  Confidence         : {result['confidence']:.0%}{review_flag}\n"
            f"  Priority tag       : {result['priority']}"
        )


# ---------------------------------------------------------------------------
# BONUS: mini CLI live demo
# ---------------------------------------------------------------------------
def live_demo(model, vectorizer):
    print("\n" + "=" * 70)
    print("LIVE DEMO - type a ticket subject and body to classify it instantly.")
    print("Type 'quit' at any prompt to exit.")
    print("=" * 70)
    while True:
        subject = input("\nSubject: ").strip()
        if subject.lower() == "quit":
            break
        body = input("Body: ").strip()
        if body.lower() == "quit":
            break
        result = predict_ticket(model, vectorizer, subject, body)
        review_flag = "  <-- NEEDS HUMAN REVIEW (low confidence)" if result["needs_human_review"] else ""
        print(
            f"  -> Category: {result['predicted_category']}  "
            f"| Confidence: {result['confidence']:.0%}{review_flag}  "
            f"| Priority: {result['priority']}"
        )


REFLECTION_NOTE = """
REFLECTION NOTE
----------------
With more data, I'd want at least a few hundred examples per category
(this demo uses ~10 per class, which is enough to prove the pipeline works
but too small to trust the accuracy number). More real historical tickets
would also expose messier language (typos, mixed-language text, forwarded
email chains) that this clean synthetic set doesn't have. With more time,
I'd add stratified k-fold cross-validation instead of a single train/test
split, try both Naive Bayes and Logistic Regression side by side with a
proper comparison table, and add a fallback "Unclear/Needs Review" label
for tickets that genuinely straddle two categories rather than forcing a
single best guess. I'd also log misclassified examples to see if specific
words are misleading the model (e.g. "account" appearing in both Billing
and HR tickets).
"""


def main():
    print("Loading and preparing data...")
    df = load_and_prepare_data(DATA_PATH)
    print(f"Loaded {len(df)} tickets across categories: {sorted(df['category'].unique())}")

    print("\nTraining model (TF-IDF + Logistic Regression)...")
    model, vectorizer, X_test_vec, y_test = train_model(df)

    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    evaluate_model(model, X_test_vec, y_test)

    run_predictions_on_samples(model, vectorizer)

    print(REFLECTION_NOTE)

    if "--demo" in sys.argv:
        live_demo(model, vectorizer)


if __name__ == "__main__":
    main()
