from flask import Flask, render_template, request, redirect, url_for
from config import Config
from models import db, SpamData, Model, Feedback, Admin
from ml.model import SpamClassifier
from ml.link_analyzer import extract_and_analyze_links, extract_urls
import joblib, io, os

from routes.admin import admin_bp
from routes.auth import auth_bp

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

classifier = SpamClassifier()

with app.app_context():
    db.create_all()

    if not Admin.query.first():
        default_admin = Admin(username='admin')
        default_admin.set_password('admin123')
        db.session.add(default_admin)
        db.session.commit()
        print("✅ Default admin created: username=admin, password=admin123")
        print("   ⚠️  Change this password after first login!")

    # ── Load active model from database ─────────────────────────────────
    # We first check if any model is marked as active in the DB.
    # If yes, load it directly from the stored pickle bytes — no file needed.
    # If no active model in DB, fall back to model.pkl file on disk.
    active_model = Model.query.filter_by(is_active=True).first()
    if active_model:
        model_bytes = io.BytesIO(active_model.pickle_data)
        classifier.model, classifier.vectorizer = joblib.load(model_bytes)
        print(f"✅ Loaded active model: {active_model.version}")
    elif os.path.exists('model.pkl'):
        classifier.load('model.pkl')
        print("✅ Loaded model from model.pkl")
    else:
        print("⚠️  No model found. Run train_model.py first.")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text'].strip()

    link_features = extract_and_analyze_links(text)
    raw_urls      = extract_urls(text)
    processed_text = text + " " + link_features

    pred, confidence = classifier.predict(processed_text, link_features)

    entry = SpamData(
        text=text,
        links=raw_urls,
        prediction=pred,
        confidence=confidence
    )
    db.session.add(entry)
    db.session.commit()

    return render_template('result.html',
                           prediction='Spam' if pred else 'Ham',
                           confidence=round(confidence * 100, 1),
                           entry_id=entry.id,
                           flagged_links=raw_urls,
                           verdict_color='#ff4d6d' if pred else '#00e096',
                           confidence_class='spam' if pred else 'ham')


@app.route('/feedback', methods=['POST'])
def feedback():
    suggestion   = request.form.get('suggestion')
    rating       = request.form.get('rating')
    spam_data_id = request.form.get('entry_id')
    if suggestion:
        fb = Feedback(suggestion=suggestion, rating=rating, spam_data_id=spam_data_id)
        db.session.add(fb)
        db.session.commit()
    return redirect(url_for('index'))


app.register_blueprint(admin_bp)
app.register_blueprint(auth_bp)

if __name__ == '__main__':
    app.run(debug=True)