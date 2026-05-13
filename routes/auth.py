from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
from models import db, Admin
import random, string, time
from functools import wraps

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

# In-memory OTP store: { email: { 'code': '123456', 'expires': timestamp } }
# Fine for a single-admin app; no external dependency needed.
_otp_store = {}

OTP_EXPIRY_SECONDS = 600  # 10 minutes


# ── Decorator ────────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated


# ── Login / Logout ────────────────────────────────────────────────────────────
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        admin    = Admin.query.filter_by(username=username).first()
        if admin and admin.check_password(password):
            session['admin_logged_in'] = True
            return redirect(url_for('admin.admin_dashboard'))
        flash('Invalid username or password.', 'danger')
    return render_template('login.html')


@auth_bp.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('index'))


# ── OTP: Request ─────────────────────────────────────────────────────────────
@auth_bp.route('/request-otp', methods=['POST'])
def request_otp():
    data  = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()

    if not email:
        return jsonify({'error': 'Email is required.'}), 400

    # Check the email belongs to a real admin account
    admin = Admin.query.filter_by(email=email).first()
    if not admin:
        # Return a vague error so we don't leak which emails exist
        return jsonify({'error': 'No admin account found with that email.'}), 404

    # Generate a 6-digit code
    code = ''.join(random.choices(string.digits, k=6))
    _otp_store[email] = {'code': code, 'expires': time.time() + OTP_EXPIRY_SECONDS}

    # Send the email
    try:
        _send_otp_email(email, code)
    except Exception as e:
        return jsonify({'error': f'Failed to send email: {str(e)}'}), 500

    return jsonify({'message': 'OTP sent successfully.'}), 200


# ── OTP: Verify ───────────────────────────────────────────────────────────────
@auth_bp.route('/verify-otp', methods=['POST'])
def verify_otp():
    data  = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    code  = (data.get('code')  or '').strip()

    entry = _otp_store.get(email)
    if not entry:
        return jsonify({'error': 'No OTP request found for this email.'}), 400
    if time.time() > entry['expires']:
        _otp_store.pop(email, None)
        return jsonify({'error': 'OTP has expired. Please request a new one.'}), 400
    if entry['code'] != code:
        return jsonify({'error': 'Incorrect code. Please try again.'}), 400

    # Mark as verified (keeps in store so reset-password can check it)
    entry['verified'] = True
    return jsonify({'message': 'OTP verified.'}), 200


# ── OTP: Reset password ───────────────────────────────────────────────────────
@auth_bp.route('/reset-password', methods=['POST'])
def reset_password():
    data         = request.get_json(silent=True) or {}
    email        = (data.get('email')        or '').strip().lower()
    code         = (data.get('code')         or '').strip()
    new_password = (data.get('new_password') or '').strip()

    if len(new_password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters.'}), 400

    entry = _otp_store.get(email)
    if not entry or not entry.get('verified') or entry['code'] != code:
        return jsonify({'error': 'OTP verification required before resetting password.'}), 400
    if time.time() > entry['expires']:
        _otp_store.pop(email, None)
        return jsonify({'error': 'Session expired. Please start over.'}), 400

    admin = Admin.query.filter_by(email=email).first()
    if not admin:
        return jsonify({'error': 'Admin account not found.'}), 404

    admin.set_password(new_password)
    db.session.commit()
    _otp_store.pop(email, None)   # Invalidate the OTP

    return jsonify({'message': 'Password updated successfully.'}), 200


# ── Email helper ──────────────────────────────────────────────────────────────
def _send_otp_email(to_email, code):
    """
    Sends the OTP via Flask-Mail.
    Make sure these env vars are set in your .env / Vercel environment:
        MAIL_USERNAME   — your Gmail / SMTP address
        MAIL_PASSWORD   — your Gmail App Password (not your normal password)
        ADMIN_EMAIL     — same as MAIL_USERNAME usually
    """
    from flask_mail import Message
    from app import mail   # imported here to avoid circular import at module level
    msg = Message(
        subject = 'SpamGuard — Your Password Reset Code',
        sender  = mail.default_sender,
        recipients = [to_email]
    )
    msg.body = (
        f"Your SpamGuard admin password reset code is:\n\n"
        f"  {code}\n\n"
        f"This code expires in 10 minutes.\n"
        f"If you did not request this, ignore this email."
    )
    msg.html = f"""
    <div style="font-family: monospace; background: #080e1a; color: #e8f0fe;
                padding: 2rem; border-radius: 12px; max-width: 420px; margin: auto;">
      <h2 style="color: #00e5ff; letter-spacing: 3px; font-size: 1.1rem;">
        SPAMGUARD — PASSWORD RESET
      </h2>
      <p style="color: #607080; font-size: 0.9rem; margin: 1rem 0 0.5rem;">
        Your one-time code:
      </p>
      <div style="font-size: 2.5rem; letter-spacing: 12px; color: #00e5ff;
                  background: #111d2e; padding: 1rem 1.5rem; border-radius: 8px;
                  border: 1px solid #1e3048; display: inline-block; margin: 0.5rem 0;">
        {code}
      </div>
      <p style="color: #607080; font-size: 0.82rem; margin-top: 1.2rem;">
        Expires in <strong style="color: #e8f0fe;">10 minutes</strong>.
        If you did not request this, ignore this email.
      </p>
    </div>
    """
    mail.send(msg)