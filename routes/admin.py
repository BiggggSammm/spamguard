from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from functools import wraps
from models import db, Model, Admin
from ml.model import SpamClassifier
from ml.link_analyzer import extract_and_analyze_links
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib, io, os

admin_bp = Blueprint('admin', __name__)


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash('Please log in to access the admin dashboard.', 'warning')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


@admin_bp.route('/admin')
@login_required
def admin_dashboard():
    models = Model.query.order_by(Model.trained_on.desc()).all()
    return render_template('admin.html', models=models)


@admin_bp.route('/admin/activate/<int:model_id>', methods=['POST'])
@login_required
def activate_model(model_id):
    model_to_activate = Model.query.get_or_404(model_id)
    Model.query.update({'is_active': False})
    model_to_activate.is_active = True
    db.session.commit()

    from app import classifier
    classifier.model, classifier.vectorizer = joblib.load(io.BytesIO(model_to_activate.pickle_data))
    classifier.save('model.pkl')

    flash(f'Model {model_to_activate.version} is now active!', 'success')
    return redirect(url_for('admin.admin_dashboard'))


@admin_bp.route('/admin/retrain', methods=['POST'])
@login_required
def retrain():
    if 'file' not in request.files or request.files['file'].filename == '':
        flash('No file selected.', 'danger')
        return redirect(url_for('admin.admin_dashboard'))

    file = request.files['file']
    os.makedirs('data', exist_ok=True)
    filepath = os.path.join('data', file.filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
        df = df.rename(columns={
            'Message': 'text', 'Spam/Ham': 'label',
            'v2': 'text', 'v1': 'label',
            'URL': 'text', 'Label': 'label'
        })

        if 'text' not in df.columns or 'label' not in df.columns:
            flash('CSV must have "text" and "label" columns.', 'danger')
            return redirect(url_for('admin.admin_dashboard'))

        df['label'] = df['label'].map({'ham': 0, 'Ham': 0, 'spam': 1, 'Spam': 1, 'good': 0, 'bad': 1})
        df = df[['text', 'label']].dropna()

        df['link_features'] = df['text'].apply(extract_and_analyze_links)
        df['combined_text'] = df['text'] + ' ' + df['link_features']

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df['combined_text'])
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        clf = SpamClassifier()
        clf.vectorizer = vectorizer
        clf.model = MultinomialNB()
        clf.model.fit(X_train_res, y_train_res)

        y_pred    = clf.model.predict(X_test)
        accuracy  = accuracy_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall    = recall_score(y_test, y_pred, average='weighted')

        last_model = Model.query.order_by(Model.id.desc()).first()
        try:
            num = float(last_model.version[1:]) if last_model else 0.9
            new_version = f"v{num + 0.1:.1f}"
        except:
            new_version = "v1.0"

        model_filename = f'model_{new_version}.pkl'
        clf.save(model_filename)

        new_model = Model(
            version=new_version,
            accuracy=round(accuracy, 4),
            f1_score=round(f1, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            is_active=False,
            pickle_data=open(model_filename, 'rb').read(),
            dataset_size=len(df)
        )
        db.session.add(new_model)
        db.session.commit()

        flash(
            f'Model {new_version} trained — Accuracy: {accuracy:.2%} | '
            f'F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}. '
            f'Click Activate to make it live.',
            'success'
        )

    except Exception as e:
        flash(f'Retraining failed: {str(e)}', 'danger')

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return redirect(url_for('admin.admin_dashboard'))

@admin_bp.route('/change-username', methods=['POST'])
@login_required
def change_username():
    data         = request.get_json(silent=True) or {}
    new_username = (data.get('new_username') or '').strip()
    password     = (data.get('password')     or '')

    if not new_username:
        return jsonify({'error': 'New username is required.'}), 400

    admin = Admin.query.first()
    if not admin.check_password(password):
        return jsonify({'error': 'Incorrect current password.'}), 403

    # Check the new username isn't already taken
    if Admin.query.filter_by(username=new_username).first():
        return jsonify({'error': 'That username is already in use.'}), 409

    admin.username = new_username
    db.session.commit()
    session.pop('admin_logged_in', None)  # force re-login with new username
    return jsonify({'message': 'Username updated successfully.'}), 200


@admin_bp.route('/change-password', methods=['POST'])
@login_required
def change_password():
    data             = request.get_json(silent=True) or {}
    current_password = (data.get('current_password') or '')
    new_password     = (data.get('new_password')     or '')

    if len(new_password) < 8:
        return jsonify({'error': 'New password must be at least 8 characters.'}), 400

    admin = Admin.query.first()
    if not admin.check_password(current_password):
        return jsonify({'error': 'Incorrect current password.'}), 403

    admin.set_password(new_password)
    db.session.commit()
    return jsonify({'message': 'Password updated successfully.'}), 200