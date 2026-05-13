# train_model.py
from app import db, app
from models import Model
from ml.model import SpamClassifier
from ml.link_analyzer import extract_and_analyze_links
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)

with app.app_context():
    print("=" * 55)
    print("   SPAMGUARD — MODEL TRAINING & EVALUATION")
    print("=" * 55)

    # ── 1. Load datasets ──────────────────────────────────────
    print("\n📂 Loading Enron email dataset...")
    from datasets import load_dataset as _load
    _enron = _load("SetFit/enron_spam")
    df_enron = _enron['train'].to_pandas()[['text', 'label']].dropna()
    df_enron['label'] = df_enron['label'].astype(int)
    print(f"   Loaded {len(df_enron)} Enron samples")

    print("\n📱 Loading SMS spam dataset...")
    df_sms = pd.read_csv('data/sms_spam.csv', encoding='latin-1')
    df_sms = df_sms[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
    df_sms['label'] = df_sms['label'].map({'ham': 0, 'spam': 1})
    df_sms = df_sms[['text', 'label']].dropna()
    print(f"   Loaded {len(df_sms)} SMS samples")

    print("\n🔗 Loading phishing email dataset from HuggingFace...")
    _phish = _load("zefang-liu/phishing-email-dataset")
    df_phish = _phish['train'].to_pandas()
    df_phish = df_phish.rename(columns={'Email Text': 'text', 'Email Type': 'label'})
    df_phish['label'] = df_phish['label'].map({'Safe Email': 0, 'Phishing Email': 1})
    df_phish = df_phish[['text', 'label']].dropna()
    # Sample to keep balanced
    df_phish = df_phish.groupby('label', group_keys=False).apply(
        lambda x: x.sample(min(len(x), 5000), random_state=42)
    ).reset_index(drop=True)
    print(f"   Loaded {len(df_phish)} phishing email samples")

    df = pd.concat([df_enron, df_sms, df_phish], ignore_index=True)

    # ── 2. Add manual examples (including many ham with trusted domains) ──
    print("\n➕ Adding manual spam/ham examples...")
    manual = pd.DataFrame({
        'text': [
            # ── Spam (14) ──────────────────────────────────────────
            "Upgrade Now for More Data! Get 5.0GB for just N1000. Dial *121*1# today!",
            "Win a free iPhone now! Click here http://bit.ly/freeiphone",
            "Congratulations! You won 1 million dollars. Claim here",
            "Limited time offer! Get 50% off today only http://offer.com",
            "Your account has been suspended. Verify now http://suspicious-link.com",
            "WINNER!! As a valued network customer you have been selected to receive a prize",
            "Click here to claim your free gift http://tinyurl.com/freegift",
            "Your PayPal account has been limited. Login at http://paypa1.com/verify",
            "Claim your reward at http://192.168.1.1/prize",
            "Free money! Visit http://cash4u.tk now",
            "URGENT: Your account will be closed. http://secure-login.verify-now.tk",
            "Dear customer, you have won a N500,000 airtime prize. Call 0812345678 to claim",
            "MTN: You have won a special promo. Reply YES to claim your N100,000 prize",
            "FCMB: Your BVN has been suspended. Click http://fcmb-verify.tk to reactivate",
            # ── Ham (20) – all contain trusted domains ──────────────
            "Check out this article https://www.google.com/search?q=python",
            "Watch the tutorial at https://www.youtube.com/watch?v=abc123",
            "Join us on https://www.linkedin.com/in/profile",
            "The documentation is at https://github.com/openai/gpt",
            "Visit our website at www.google.com for more information",
            "Here is the link: https://www.wikipedia.org/wiki/Python",
            "Sign in at https://www.paypal.com to check your balance",
            "Download from https://www.microsoft.com/downloads",
            "More info at https://stackoverflow.com/questions/12345",
            "See https://www.reddit.com/r/python for discussion",
            "Hi, please check the meeting notes at https://docs.google.com",
            "Your receipt is at https://www.amazon.com/orders",
            "Shop the latest deals at https://www.jumia.com.ng",
            "Your order from https://www.jumia.com.ng has been confirmed",
            "Check the latest phones at https://www.jumia.com.ng/phones",
            "This week on Substack: https://substack.com/inbox",
            "Read the full newsletter at https://techcrunch.substack.com",
            "Meeting link: https://zoom.us/j/123456789",
            "Here is the Slack channel: https://slack.com/app",
            "See the report at https://drive.google.com/file/abc",
        ],
        'label': [
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,  # 14 spam
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0   # 20 ham
        ]
    })

    # ── 3. Merge ──────────────────────────────────────────────
    print("\n🔀 Merging datasets...")
    df = pd.concat([df, manual], ignore_index=True)
    df = df[['text', 'label']].dropna()
    print(f"   Total samples: {len(df)}")

    spam_count = int(df['label'].sum())
    ham_count  = len(df) - spam_count
    print(f"\n📊 Class Distribution:")
    print(f"   Ham  (0): {ham_count} ({ham_count/len(df)*100:.1f}%)")
    print(f"   Spam (1): {spam_count} ({spam_count/len(df)*100:.1f}%)")

    # ── 4. Feature extraction ─────────────────────────────────
    print("\n🔗 Extracting link features...")
    df['link_features'] = df['text'].apply(extract_and_analyze_links)
    # IMPORTANT: Do NOT strip URLs from the text – we want the model to see domain names.
    # The link_features already contain structural flags; keeping URLs in text allows TF‑IDF
    # to learn that tokens like 'google.com' are strong ham indicators.
    df['combined_text'] = df['text'] + " LINK_FEATURES: " + df['link_features']

    print("📝 Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 3))
    X = vectorizer.fit_transform(df['combined_text'])
    y = df['label']

    # ── 5. Train / test split ─────────────────────────────────
    print("\n✂️  Splitting: 80% train / 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # ── 6. Train ──────────────────────────────────────────────
    print("\n🤖 Training Naive Bayes classifier...")
    classifier         = SpamClassifier()
    classifier.vectorizer = vectorizer
    classifier.model   = MultinomialNB(alpha=0.1)
    classifier.model.fit(X_train, y_train)
    print("   Training complete!")

    # ── 7. Evaluate ───────────────────────────────────────────
    print("\n📈 Evaluating on test set...")
    y_pred    = classifier.model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall    = recall_score(y_test, y_pred, average='weighted')
    f1        = f1_score(y_test, y_pred, average='weighted')

    print("\n" + "=" * 55)
    print("   EVALUATION RESULTS")
    print("=" * 55)
    print(f"   Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"   Precision : {precision:.4f}")
    print(f"   Recall    : {recall:.4f}")
    print(f"   F1-Score  : {f1:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    cm = confusion_matrix(y_test, y_pred)
    print("🔢 Confusion Matrix:")
    print(f"   TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"   FN={cm[1][0]}  TP={cm[1][1]}")
    print()
    print("   TN = Ham correctly identified as Ham")
    print("   FP = Ham wrongly flagged as Spam  ← want this low")
    print("   FN = Spam that slipped through    ← want this low")
    print("   TP = Spam correctly caught")

    # ── 8. Save ───────────────────────────────────────────────
    last_model = Model.query.order_by(Model.id.desc()).first()
    if last_model and last_model.version.startswith('v'):
        try:
            new_version = f"v{float(last_model.version[1:]) + 0.1:.1f}"
        except:
            new_version = "v1.0"
    else:
        new_version = "v1.0"

    classifier.save('model.pkl')

    with open('model.pkl', 'rb') as f:
        pickle_bytes = f.read()

    db.session.add(Model(
        version=new_version,
        accuracy=round(accuracy, 4),
        f1_score=round(f1, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        is_active=False,
        pickle_data=pickle_bytes,
        dataset_size=len(df)
    ))
    db.session.commit()

    print("\n" + "=" * 55)
    print(f"   ✅ Model {new_version} saved!")
    print(f"   Total samples: {len(df)}")
    print("   Go to admin dashboard and click ACTIVATE to go live.")
    print("=" * 55)