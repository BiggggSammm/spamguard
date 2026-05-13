import os
import sys
import io
import re
import joblib
from flask import Flask, render_template, request, redirect, url_for
from dotenv import load_dotenv

# Add parent directory so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import db, SpamData, Model, Feedback, Admin
from ml.model import SpamClassifier
from ml.link_analyzer import check_links_for_decision, analyze_links_for_prediction, extract_urls
from routes.admin import admin_bp
from routes.auth import auth_bp

load_dotenv()   # only for local development

app = Flask(__name__, static_folder='../static', template_folder='../templates')

# Configuration – must exist
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-me')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
if not app.config['SQLALCHEMY_DATABASE_URI']:
    raise ValueError("DATABASE_URL environment variable not set")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

classifier = SpamClassifier()

def load_active_model():
    """Load the currently active model from Supabase."""
    with app.app_context():
        try:
            active = Model.query.filter_by(is_active=True).first()
            if active and active.pickle_data:
                classifier.model, classifier.vectorizer = joblib.load(io.BytesIO(active.pickle_data))
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
    return False

# Load model on startup (Vercel does this once per cold start)
load_active_model()

# ---------- Routes ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text'].strip()
    if not text:
        return redirect(url_for('index'))

    any_suspicious, suspicious_urls, safe_urls, link_messages = check_links_for_decision(text)
    clean_text = re.sub(r'(?:https?://|www\.)\S+', '', text).strip()
    is_pure_url = (clean_text == "" and len(extract_urls(text)) > 0)

    if any_suspicious:
        verdict = 'Spam'
        verdict_color = '#ff4d6d'
        conf_class = 'spam'
        text_confidence = None
        pred = 1
    else:
        link_features, _ = analyze_links_for_prediction(text)
        processed_text = clean_text + ' ' + link_features
        pred, conf = classifier.predict(processed_text, link_features)
        text_confidence = conf
        if pred == 1:
            verdict = 'Spam'
            verdict_color = '#ff4d6d'
            conf_class = 'spam'
        else:
            verdict = 'Ham'
            verdict_color = '#00e096'
            conf_class = 'ham'

    if is_pure_url:
        text_confidence = None

    entry = SpamData(
        text=text,
        links=suspicious_urls if suspicious_urls else None,
        prediction=pred,
        confidence=text_confidence if text_confidence is not None else 0.99
    )
    db.session.add(entry)
    db.session.commit()

    return render_template('result.html',
                           verdict=verdict,
                           text_confidence=text_confidence,
                           entry_id=entry.id,
                           flagged_links=suspicious_urls,
                           verdict_color=verdict_color,
                           confidence_class=conf_class,
                           link_analysis=link_messages,
                           is_pure_url=is_pure_url)

@app.route('/feedback', methods=['POST'])
def feedback():
    suggestion = request.form.get('suggestion')
    rating = request.form.get('rating')
    spam_data_id = request.form.get('entry_id')
    if suggestion:
        fb = Feedback(suggestion=suggestion, rating=rating, spam_data_id=spam_data_id)
        db.session.add(fb)
        db.session.commit()
    return redirect(url_for('index'))

# ----- Admin endpoint for uploading a retrained model -----
@admin_bp.route('/upload_model', methods=['POST'])
def upload_model():
    """Admin uploads a new model file (pickle) and saves it to Supabase."""
    if 'model_file' not in request.files:
        return "No file part", 400
    file = request.files['model_file']
    if file.filename == '':
        return "No selected file", 400
    if file and file.filename.endswith('.pkl'):
        model_bytes = file.read()
        # Deactivate current active model
        Model.query.update({Model.is_active: False})
        # Store new model
        new_model = Model(
            version=f"v{Model.query.count() + 1}.0",
            accuracy=0.0,
            pickle_data=model_bytes,
            is_active=True,
            dataset_size=0
        )
        db.session.add(new_model)
        db.session.commit()
        # Reload classifier with new model
        load_active_model()
        # Redirect to admin dashboard (adjust if your dashboard route is different)
        return redirect(url_for('admin.dashboard'))
    return "Invalid file", 400

# Register blueprints
app.register_blueprint(admin_bp)
app.register_blueprint(auth_bp)

# For local testing only – Vercel ignores this
if __name__ == '__main__':
    app.run(debug=False)