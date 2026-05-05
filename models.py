from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

# ─────────────────────────────────────────────
# Admin user table
# ─────────────────────────────────────────────
class Admin(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

    def set_password(self, password):
        """Hash and store the password — we never store plain text."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Return True if the given password matches the stored hash."""
        return check_password_hash(self.password_hash, password)

# ─────────────────────────────────────────────
# Spam detection log table
# ─────────────────────────────────────────────
class SpamData(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    text       = db.Column(db.Text, nullable=False)
    links      = db.Column(db.JSON)
    prediction = db.Column(db.Integer)      # 0 = ham, 1 = spam
    confidence = db.Column(db.Float)
    label      = db.Column(db.Integer)
    timestamp  = db.Column(db.DateTime, default=datetime.utcnow)
    model_id   = db.Column(db.Integer, db.ForeignKey('model.id'))

# ─────────────────────────────────────────────
# Trained model versions table
# ─────────────────────────────────────────────
class Model(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    version      = db.Column(db.String(50), unique=True, nullable=False)
    accuracy     = db.Column(db.Float)
    f1_score     = db.Column(db.Float)
    precision    = db.Column(db.Float)           # how many flagged were actually spam
    recall       = db.Column(db.Float)           # how many actual spam were caught
    is_active    = db.Column(db.Boolean, default=False)  # which model is currently live
    pickle_data  = db.Column(db.LargeBinary)
    trained_on   = db.Column(db.DateTime, default=datetime.utcnow)
    dataset_size = db.Column(db.Integer)

# ─────────────────────────────────────────────
# User feedback table
# ─────────────────────────────────────────────
class Feedback(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    suggestion   = db.Column(db.Text)
    rating       = db.Column(db.Integer)
    timestamp    = db.Column(db.DateTime, default=datetime.utcnow)
    spam_data_id = db.Column(db.Integer, db.ForeignKey('spam_data.id'))