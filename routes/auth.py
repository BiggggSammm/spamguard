# routes/auth.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from models import Admin

auth_bp = Blueprint('auth', __name__)

# ─────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    # If already logged in, go straight to admin dashboard
    if session.get('admin_logged_in'):
        return redirect(url_for('admin.admin_dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        # Look up the admin by username
        admin = Admin.query.filter_by(username=username).first()

        # check_password() compares against the stored hash — never plain text
        if admin and admin.check_password(password):
            session['admin_logged_in'] = True   # mark this browser session as authenticated
            session['admin_username'] = username
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('admin.admin_dashboard'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html')

# ─────────────────────────────────────────────
# LOGOUT
# ─────────────────────────────────────────────
@auth_bp.route('/logout')
def logout():
    session.clear()   # wipe everything from the session
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))