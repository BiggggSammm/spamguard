# train_model.py
from app import db, app
from models import Model
from ml.model import SpamClassifier
from ml.link_analyzer import extract_and_analyze_links
from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

with app.app_context():
    print("=" * 55)
    print("   SPAM DETECTION MODEL — TRAINING & EVALUATION")
    print("=" * 55)

    # ── 1. Load Enron Email Dataset ───────────────────────────
    print("\n📂 Loading Enron spam dataset from Hugging Face...")
    dataset = load_dataset("SetFit/enron_spam")
    df_enron = dataset['train'].to_pandas()
    df_enron = df_enron.rename(columns={'text': 'text', 'label': 'label'})
    df_enron = df_enron[['text', 'label']].dropna()
    print(f"   Loaded {len(df_enron)} samples from Enron dataset")

    # ── 2. Load SMS Spam Dataset ──────────────────────────────
    # sms_spam.csv has columns v1 (ham/spam label) and v2 (message text)
    print("\n📱 Loading SMS spam dataset from data/sms_spam.csv...")
    df_sms = pd.read_csv('data/sms_spam.csv', encoding='latin-1')
    df_sms = df_sms[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
    df_sms['label'] = df_sms['label'].map({'ham': 0, 'spam': 1})
    df_sms = df_sms[['text', 'label']].dropna()
    print(f"   Loaded {len(df_sms)} samples from SMS dataset")

    # ── 3. Load Phishing URL Dataset ──────────────────────────
    # phishing_urls.csv has URL and structural features.
    # We convert each URL into a descriptive text string so the
    # Naive Bayes classifier can learn from URL-based patterns.
    # label: 1 = phishing, 0 = legitimate
    # ── 3. Load Phishing URL Dataset ──────────────────────────
    print("\n🔗 Loading phishing URL dataset from data/phishing_urls.csv...")
    df_url = pd.read_csv('data/phishing_urls.csv', encoding='utf-8-sig')
    df_url.columns = df_url.columns.str.strip()

    # Stratified sample that keeps the label column intact
    df_url = df_url.sample(frac=1, random_state=42).groupby('label').head(2000).reset_index(drop=True)

    print(f"   Loaded {len(df_url)} samples after sampling")

    def url_to_text(row):
        url = row.get('URL', '') or row.get('FILENAME', '')
        tokens = [f"url {url}"]
        if row.get('IsDomainIP', 0) == 1: tokens.append('domain_ip')
        if row.get('NoOfSubDomain', 0) > 2: tokens.append('many_subdomains')
        if row.get('HasObfuscation', 0) == 1: tokens.append('obfuscated_url')
        if row.get('IsHTTPS', 0) == 0: tokens.append('not_https')
        if row.get('URLLength', 0) > 75: tokens.append('long_url')
        if row.get('NoOfURLRedirect', 0) > 0: tokens.append('has_redirect')
        if row.get('HasPasswordField', 0) == 1: tokens.append('has_password_field')
        if row.get('Bank', 0) == 1: tokens.append('bank_related')
        if row.get('Pay', 0) == 1: tokens.append('payment_related')
        return ' '.join(tokens)

    df_url['text'] = df_url.apply(url_to_text, axis=1)
    df_url['label'] = df_url['label'].astype(int)
    df_url = df_url[['text', 'label']].dropna()
    print(f"   Final phishing samples: {len(df_url)}")    
    # ── 4. Add Manual Promo/Phishing Examples ─────────────────
    promo = pd.DataFrame({
        'text': [
            "Upgrade Now for More Data! Get 5.0GB for just N1000. Dial *121*1# today!",
            "Win a free iPhone now! Click here http://bit.ly/freeiphone",
            "Congratulations! You won 1 million dollars. Claim here",
            "Limited time offer! Get 50% off today only http://offer.com",
            "Your account has been suspended. Verify now http://suspicious-link.com",
            "WINNER!! As a valued network customer you have been selected to receive a prize",
            "Click here to claim your free gift http://tinyurl.com/freegift",
            "Your PayPal account has been limited. Login at http://paypa1.com/verify"
        ],
        'label': [1, 1, 1, 1, 1, 1, 1, 1]
    })

    # ── 5. Merge All Datasets ─────────────────────────────────
    print("\n🔀 Merging all datasets...")
    df = pd.concat([df_enron, df_sms, df_url, promo], ignore_index=True)
    df = df[['text', 'label']].dropna()
    print(f"   Total samples: {len(df)}")

    # ── 6. Class Distribution ─────────────────────────────────
    spam_count = int(df['label'].sum())
    ham_count  = len(df) - spam_count
    print(f"\n📊 Class Distribution:")
    print(f"   Ham  (0): {ham_count} samples ({ham_count/len(df)*100:.1f}%)")
    print(f"   Spam (1): {spam_count} samples ({spam_count/len(df)*100:.1f}%)")

    # ── 7. Feature Extraction ─────────────────────────────────
    print("\n🔗 Extracting link features from messages...")
    df['link_features'] = df['text'].apply(extract_and_analyze_links)
    df['combined_text'] = df['text'] + " LINK_FEATURES: " + df['link_features']

    print("📝 Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 3))
    X = vectorizer.fit_transform(df['combined_text'])
    y = df['label']

    # ── 8. Train / Test Split ─────────────────────────────────
    # We split BEFORE training so the model never sees test data during training.
    # test_size=0.2 means 80% trains the model, 20% is held back to evaluate it.
    # random_state=42 ensures the same split every time you run (reproducibility).
    # stratify=y ensures both splits have the same ham/spam ratio.
    print("\n✂️  Splitting dataset: 80% train / 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training samples : {X_train.shape[0]}")
    print(f"   Test samples     : {X_test.shape[0]}")

    # ── 9. Train the Model ────────────────────────────────────
    # alpha=0.1 is Laplace smoothing — prevents zero probabilities for unseen words
    # class_prior=[0.3, 0.7] tells the model to lean toward spam since phishing
    # URLs make spam slightly more represented in our merged dataset
    print("\n🤖 Training Naive Bayes classifier...")
    classifier = SpamClassifier()
    classifier.vectorizer = vectorizer
    classifier.model = MultinomialNB(alpha=0.1)
    classifier.model.fit(X_train, y_train)
    print("   Training complete!")

    # ── 10. Evaluate on Unseen Test Set ──────────────────────
    # All metrics below are computed on X_test — data the model has NEVER seen.
    # This gives us an honest picture of real-world performance.
    print("\n📈 Evaluating on test set...")
    y_pred = classifier.model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall    = recall_score(y_test, y_pred, average='weighted')
    f1        = f1_score(y_test, y_pred, average='weighted')

    # ── 11. Print Full Report ─────────────────────────────────
    print("\n" + "=" * 55)
    print("   EVALUATION RESULTS")
    print("=" * 55)
    print(f"   Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"   Precision : {precision:.4f}")
    print(f"   Recall    : {recall:.4f}")
    print(f"   F1-Score  : {f1:.4f}")

    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    print("🔢 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   [[TN={cm[0][0]}  FP={cm[0][1]}]")
    print(f"    [FN={cm[1][0]}  TP={cm[1][1]}]]")
    print()
    print("   TN = Ham correctly identified as Ham")
    print("   FP = Ham wrongly flagged as Spam")
    print("   FN = Spam that slipped through as Ham")
    print("   TP = Spam correctly caught")

    # ── 12. Auto-generate version number ─────────────────────
    # Checks the DB for the latest version and increments by 0.1
    # so we never get a UNIQUE constraint error even if you run
    # this script multiple times.
    last_model = Model.query.order_by(Model.id.desc()).first()
    if last_model and last_model.version.startswith('v'):
        try:
            num = float(last_model.version[1:])
            new_version = f"v{num + 0.1:.1f}"
        except:
            new_version = "v1.0"
    else:
        new_version = "v1.0"

    # ── 13. Save Model ────────────────────────────────────────
    classifier.save('model.pkl')

    new_model = Model(
        version=new_version,
        accuracy=round(accuracy, 4),
        f1_score=round(f1, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        is_active=False,    # not active until admin clicks Activate
        pickle_data=open('model.pkl', 'rb').read(),
        dataset_size=len(df)
    )
    db.session.add(new_model)
    db.session.commit()

    print("\n" + "=" * 55)
    print(f"   ✅ Model {new_version} saved successfully!")
    print(f"   Datasets used:")
    print(f"   - Enron emails  : {len(df_enron)} samples")
    print(f"   - SMS messages  : {len(df_sms)} samples")
    print(f"   - Phishing URLs : {len(df_url)} samples")
    print(f"   - Manual promo  : {len(promo)} samples")
    print(f"   - Total         : {len(df)} samples")
    print("   Go to the admin dashboard and click ACTIVATE to make it live.")
    print("=" * 55)