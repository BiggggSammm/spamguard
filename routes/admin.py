# routes/admin.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from functools import wraps
from models import db, Model
from ml.model import SpamClassifier
from ml.link_analyzer import extract_and_analyze_links
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib, io, os

admin_bp = Blueprint('admin', __name__)

# ─────────────────────────────────────────────
# Login-required decorator
# ─────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash('Please log in to access the admin dashboard.', 'warning')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


# ─────────────────────────────────────────────
# Admin Dashboard
# ─────────────────────────────────────────────
@admin_bp.route('/admin')
@login_required
def admin_dashboard():
    models = Model.query.order_by(Model.trained_on.desc()).all()
    return render_template('admin.html', models=models)


# ─────────────────────────────────────────────
# Activate a model version
# ─────────────────────────────────────────────
@admin_bp.route('/admin/activate/<int:model_id>', methods=['POST'])
@login_required
def activate_model(model_id):
    # Step 1 — find the model we want to activate
    model_to_activate = Model.query.get_or_404(model_id)

    # Step 2 — deactivate ALL other models first
    # Only one model can be active at a time
    Model.query.update({'is_active': False})

    # Step 3 — mark the selected one as active
    model_to_activate.is_active = True
    db.session.commit()

    # Step 4 — load the pickle data from the DB into the running classifier
    # We import app's classifier here to update it live without restarting
    from app import classifier
    model_bytes = io.BytesIO(model_to_activate.pickle_data)
    classifier.model, classifier.vectorizer = joblib.load(model_bytes)

    # Step 5 — also save it as model.pkl so it persists after restart
    classifier.save('model.pkl')

    flash(f'Model {model_to_activate.version} is now active!', 'success')
    return redirect(url_for('admin.admin_dashboard'))


# ─────────────────────────────────────────────
# Retrain route
# ─────────────────────────────────────────────
@admin_bp.route('/admin/retrain', methods=['POST'])
@login_required
def retrain():
    if 'file' not in request.files:
        flash('No file uploaded', 'danger')
        return redirect(url_for('admin.admin_dashboard'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('admin.admin_dashboard'))

    os.makedirs('data', exist_ok=True)
    filepath = os.path.join('data', file.filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
        print("Uploaded CSV columns:", list(df.columns))

        df = df.rename(columns={
            'Message': 'text', 'Spam/Ham': 'label',
            'v2': 'text', 'v1': 'label',
            'URL': 'text', 'Label': 'label'
        })

        if 'text' not in df.columns or 'label' not in df.columns:
            flash('CSV must contain "text" and "label" columns', 'danger')
            return redirect(url_for('admin.admin_dashboard'))

        df['label'] = df['label'].map({
            'ham': 0, 'Ham': 0, 'spam': 1, 'Spam': 1, 'good': 0, 'bad': 1
        })
        df = df[['text', 'label']].dropna()

        df['link_features'] = df['text'].apply(extract_and_analyze_links)
        df['combined_text'] = df['text'] + " " + df['link_features']

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df['combined_text'])
        y = df['label']

        # 80/20 train/test split before SMOTE
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # SMOTE on training data only
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Train
        classifier = SpamClassifier()
        classifier.vectorizer = vectorizer
        classifier.model = MultinomialNB()
        classifier.model.fit(X_train_resampled, y_train_resampled)

        # ── Evaluate on unseen test set ──────────────────────────────────
        y_pred    = classifier.model.predict(X_test)
        accuracy  = classifier.model.score(X_test, y_test)
        f1        = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')  # ← new
        recall    = recall_score(y_test, y_pred, average='weighted')     # ← new

        # Generate version number
        last_model = Model.query.order_by(Model.id.desc()).first()
        if last_model and last_model.version.startswith('v'):
            try:
                num = float(last_model.version[1:])
                new_version = f"v{num + 0.1:.1f}"
            except:
                new_version = "v1.0"
        else:
            new_version = "v1.0"

        # Save pickle file
        model_filename = f'model_{new_version}.pkl'
        classifier.save(model_filename)

        # Save to database — new model is NOT active by default
        # Admin must explicitly click Activate to make it live
        new_model = Model(
            version=new_version,
            accuracy=round(accuracy, 4),
            f1_score=round(f1, 4),
            precision=round(precision, 4),   # ← new
            recall=round(recall, 4),         # ← new
            is_active=False,                 # ← not active until admin activates it
            pickle_data=open(model_filename, 'rb').read(),
            dataset_size=len(df)
        )
        db.session.add(new_model)
        db.session.commit()

        flash(
            f'Model {new_version} trained — '
            f'Accuracy: {accuracy:.2%} | F1: {f1:.4f} | '
            f'Precision: {precision:.4f} | Recall: {recall:.4f}. '
            f'Click Activate to make it live.',
            'success'
        )

    except Exception as e:
        flash(f'Error during retraining: {str(e)}', 'danger')

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return redirect(url_for('admin.admin_dashboard'))