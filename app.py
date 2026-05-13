import os
import re
import io
import joblib
from flask import Flask, render_template, request, redirect, url_for
from flask_mail import Mail
from config import Config
from models import db, SpamData, Model, Feedback, Admin
from ml.model import SpamClassifier
from ml.link_analyzer import (
    extract_and_analyze_links,
    analyze_links_for_prediction,
    extract_urls,
    check_links_for_decision
)
from routes.admin import admin_bp
from routes.auth import auth_bp

app = Flask(__name__)
app.config.from_object(Config)

os.makedirs(os.path.join(os.path.dirname(__file__), 'instance'), exist_ok=True)

db.init_app(app)
mail = Mail(app)

classifier = SpamClassifier()

with app.app_context():
    db.create_all()

    if not Admin.query.first():
        default_admin = Admin(username='admin')
        default_admin.set_password('admin123')
        default_admin.email = 'admin@example.com'
        db.session.add(default_admin)
        db.session.commit()
    else:
        existing = Admin.query.first()
        if not existing.email:
            existing.email = 'admin@example.com'
            db.session.commit()

    active_model = Model.query.filter_by(is_active=True).first()
    if active_model and active_model.pickle_data:
        classifier.model, classifier.vectorizer = joblib.load(io.BytesIO(active_model.pickle_data))
    elif os.path.exists('model.pkl'):
        classifier.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text'].strip()
    if not text:
        return redirect(url_for('index'))

    # 1. Get detailed link analysis
    any_suspicious, suspicious_urls, safe_urls, link_messages = check_links_for_decision(text)

    # 2. Determine if input is pure URL(s) only (no other text)
    clean_text_only = re.sub(r'(?:https?://|www\.)\S+', '', text).strip()
    is_pure_url_input = (clean_text_only == "" and len(extract_urls(text)) > 0)

    # 3. Rule: if any suspicious link -> immediate Spam (no text ML needed)
    if any_suspicious:
        verdict = 'Spam'
        verdict_color = '#ff4d6d'
        conf_class = 'spam'
        text_confidence = None          # no text confidence because rule-based
        pred = 1
    else:
        # No suspicious links -> always run ML on the text (even if trusted domains exist)
        # Clean text: remove URLs so TF‑IDF doesn't get confused by long random tokens
        clean_text = re.sub(r'(?:https?://|www\.)\S+', '', text).strip()
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

    # For pure URL input, we hide the text confidence bar (no text to analyse)
    if is_pure_url_input:
        text_confidence = None

    # Store record
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
                           is_pure_url=is_pure_url_input)

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

app.register_blueprint(admin_bp)
app.register_blueprint(auth_bp)

if __name__ == '__main__':
    app.run(debug=False)