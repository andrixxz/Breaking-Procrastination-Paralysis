from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import joblib
import os
from datetime import datetime, date, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# creates flask app instance
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Session configuration - ensures sessions persist across requests
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# file paths
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models") # where ML model + vectorizer are stored
DB_PATH = os.path.join(BASE_DIR, "journal.db") # SQLite database

# model file paths
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.pkl")

# load ML model + vectorizer
vectorizer = joblib.load(VECTORIZER_PATH) # converts journal text into feature vectors
emotion_model = joblib.load(MODEL_PATH) # logistic regression classifier for emotions


# database setup + connection

def get_db_connection():
    # opens DB with rows returned as dictionary-like objects
    conn = sqlite3.connect(DB_PATH, timeout=3)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    # creates database tables on first run
    conn = get_db_connection()
    cur = conn.cursor()

    # users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)

    # journal entries table (per user)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS journal_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            entry_text TEXT NOT NULL,
            predicted_emotion TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)

    # alignment score + emotional streak (per user)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS alignment_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE NOT NULL,
            alignment_score INTEGER NOT NULL,
            emotional_streak INTEGER NOT NULL,
            last_journal_date TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)

    # habit list table (per user)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS habits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            is_sample INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)

    # record of when habits are completed
    cur.execute("""
        CREATE TABLE IF NOT EXISTS habit_completions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            habit_id INTEGER NOT NULL,
            completed_at TEXT NOT NULL,
            FOREIGN KEY (habit_id) REFERENCES habits(id)
        );
    """)

    conn.commit()
    conn.close()


def initialize_user_data(user_id):
    # creates alignment state and sample habits for a new user
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO alignment_state (user_id, alignment_score, emotional_streak, last_journal_date) "
        "VALUES (?, 0, 0, NULL);",
        (user_id,),
    )

    now = datetime.now().isoformat(timespec="seconds")
    sample_habits = [
        "Write one sentence for an assignment",
        "Open my notes and read for 5 minutes",
        "Tidy my desk for two minutes",
    ]
    for name in sample_habits:
        cur.execute(
            "INSERT INTO habits (user_id, name, is_sample, created_at) VALUES (?, ?, 1, ?)",
            (user_id, name, now),
        )

    conn.commit()
    conn.close()


# authentication

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('landing'))
        return f(*args, **kwargs)
    return decorated_function


# support functions

def get_greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    elif hour < 21:
        return "Good evening"
    else:
        return "Welcome back"


def normalise_entry_text(text: str) -> str:
    # trims whitespace + capitalises first letter
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def predict_emotion(text: str) -> str:
    # convert journal entry then tf-idf features then predict using LR model
    X = vectorizer.transform([text])
    pred = emotion_model.predict(X)[0]
    return pred


def get_reframe_and_affirmation(predicted_emotion: str):
    # emotion supportive reframes + affirmations

    reframes = {
        "overwhelmed": (
            "It makes sense that you feel overwhelmed because you care and want to do well. "
            "You can take things in tiny pieces and still move forward."
        ),
        "anxious": (
            "You feel anxious because the outcome matters to you, not because you are failing. "
            "You can calm your mind and take one small step."
        ),
        "stuck": (
            "Feeling stuck doesn't mean you can't do it. "
            "It just means the first step needs to be smaller and more approachable."
        ),
        "stressed": (
            "You feel stressed because you are carrying a lot. "
            "You can release the pressure and take things one small action at a time."
        ),
        "tired": (
            "You feel tired because you have been pushing yourself mentally or emotionally. "
            "Resting can help you come back with more clarity."
        ),
        "calm": (
            "Feeling calm helps everything feel more manageable and within reach."
        ),
    }

    affirmations = {
        "overwhelmed": "I can take this slowly and still make progress.",
        "anxious": "I can steady myself and begin gently.",
        "stuck": "I move forward in small steps even when it feels hard.",
        "stressed": "I can pace myself and still get things done.",
        "tired": "I am allowed to pause without losing progress.",
        "calm": "I am becoming the version of myself I imagined.",
    }

    default_reframe = (
        "Whatever I am feeling right now is valid and I can still take one tiny step."
    )
    default_affirmation = "I am doing my best and that is enough."

    reframe = reframes.get(predicted_emotion, default_reframe)
    affirmation = affirmations.get(predicted_emotion, default_affirmation)
    return reframe, affirmation


def get_alignment_state(user_id):
    # returns alignment score + streak for a specific user
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT alignment_score, emotional_streak, last_journal_date "
        "FROM alignment_state WHERE user_id = ?;",
        (user_id,),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return 0, 0, None
    return row["alignment_score"], row["emotional_streak"], row["last_journal_date"]


def update_alignment_score(user_id, delta: int):
    # increase or reduce score but never below zero
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE alignment_state "
        "SET alignment_score = MAX(alignment_score + ?, 0) "
        "WHERE user_id = ?;",
        (delta, user_id),
    )
    conn.commit()
    conn.close()


def update_emotional_streak_for_today(user_id):
    # journals on consecutive days then streak increases, skip a day resets
    today = date.today()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT emotional_streak, last_journal_date FROM alignment_state WHERE user_id = ?;",
        (user_id,),
    )
    row = cur.fetchone()
    if row is None:
        conn.close()
        return

    streak = row["emotional_streak"]
    last_str = row["last_journal_date"]

    if last_str is None:
        # first ever journal
        streak = 1
    else:
        try:
            last_date = date.fromisoformat(last_str)
            if last_date == today:
                # already counted today, leave streak as is
                pass
            elif last_date == today - timedelta(days=1):
                # journalled yesterday, so streak continues
                streak += 1
            else:
                # gap in days, streak resets
                streak = 1
        except ValueError:
            streak = 1

    cur.execute(
        "UPDATE alignment_state SET emotional_streak = ?, last_journal_date = ? "
        "WHERE user_id = ?;",
        (streak, today.isoformat(), user_id),
    )
    conn.commit()
    conn.close()


# routes - authentication

@app.route("/")
def landing():
    # public landing page — redirect to dashboard if already logged in
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template("landing.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not password:
            flash("Username and password are required.", "error")
            return render_template("signup.html")

        if len(username) < 3:
            flash("Username must be at least 3 characters.", "error")
            return render_template("signup.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("signup.html")

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("signup.html")

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        existing = cur.fetchone()

        if existing:
            conn.close()
            flash("Username already taken. Please choose another.", "error")
            return render_template("signup.html")

        password_hash = generate_password_hash(password)
        cur.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
        user_id = cur.lastrowid
        conn.close()

        initialize_user_data(user_id)

        session.permanent = True
        session['user_id'] = user_id
        session['username'] = username
        return redirect(url_for('dashboard'))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session.permanent = True
            session['user_id'] = user["id"]
            session['username'] = username
            return redirect(url_for('dashboard'))

        flash("Invalid username or password.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('landing'))


# routes - UI screens

@app.route("/dashboard")
@login_required
def dashboard():
    # homepage showing score + streak + affirmation
    user_id = session['user_id']
    alignment_score, emotional_streak, last_journal_date = get_alignment_state(user_id)

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT entry_text, predicted_emotion, created_at "
        "FROM journal_entries WHERE user_id = ? ORDER BY id DESC LIMIT 1;",
        (user_id,),
    )
    last_entry = cur.fetchone()
    conn.close()

    today_affirmation = None
    if last_entry:
        _, emotion, _ = last_entry
        _, today_affirmation = get_reframe_and_affirmation(emotion)

    return render_template(
        "dashboard.html",
        greeting=get_greeting(),
        alignment_score=alignment_score,
        emotional_streak=emotional_streak,
        today_affirmation=today_affirmation,
    )

# main journaling page
@app.route("/journal", methods=["GET", "POST"])
@login_required
def journal():
    user_id = session['user_id']
    predicted_emotion = None
    reframe = None
    affirmation = None

    if request.method == "POST":
        raw_text = request.form.get("entry_text", "")
        entry_text = normalise_entry_text(raw_text)

        if entry_text:
            predicted_emotion = predict_emotion(entry_text)
            reframe, affirmation = get_reframe_and_affirmation(predicted_emotion)

            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, created_at) "
                "VALUES (?, ?, ?, ?)",
                (user_id, entry_text, predicted_emotion,
                 datetime.now().isoformat(timespec="seconds")),
            )
            conn.commit()
            conn.close()

            # journaling counts as identity-aligned behaviour
            update_alignment_score(user_id, 1)
            update_emotional_streak_for_today(user_id)

    # load journal history
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, entry_text, predicted_emotion, created_at "
        "FROM journal_entries WHERE user_id = ? ORDER BY id DESC;",
        (user_id,),
    )
    entries = cur.fetchall()
    conn.close()

    return render_template(
        "journal.html",
        predicted_emotion=predicted_emotion,
        reframe=reframe,
        affirmation=affirmation,
        entries=entries,
    )


@app.route("/journal/<int:entry_id>/edit", methods=["GET", "POST"])
@login_required
def edit_journal(entry_id):
    # edit an existing entry and re-predict new emotion
    user_id = session['user_id']

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, entry_text, predicted_emotion, created_at "
        "FROM journal_entries WHERE id = ? AND user_id = ?;",
        (entry_id, user_id),
    )
    entry = cur.fetchone()

    if entry is None:
        conn.close()
        return redirect(url_for("journal"))

    if request.method == "POST":
        raw_text = request.form.get("entry_text", "")
        entry_text = normalise_entry_text(raw_text)

        if entry_text:
            new_emotion = predict_emotion(entry_text)
            cur.execute(
                "UPDATE journal_entries "
                "SET entry_text = ?, predicted_emotion = ? "
                "WHERE id = ? AND user_id = ?;",
                (entry_text, new_emotion, entry_id, user_id),
            )
            conn.commit()

        conn.close()
        return redirect(url_for("journal"))

    conn.close()
    return render_template("edit_journal.html", entry=entry)


@app.route("/journal/<int:entry_id>/delete", methods=["POST"])
@login_required
def delete_journal(entry_id):
    # delete a journal entry and gently reduce alignment score by 1
    user_id = session['user_id']

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM journal_entries WHERE id = ? AND user_id = ?;",
        (entry_id, user_id),
    )
    conn.commit()
    conn.close()

    # deleting an entry removes one point but never below zero
    update_alignment_score(user_id, -1)

    return redirect(url_for("journal"))


@app.route("/habits", methods=["GET", "POST"])
@login_required
def habits():
    user_id = session['user_id']

    if request.method == "POST":
        # if user added a new habit/task
        new_habit = request.form.get("new_habit", "").strip()
        if new_habit:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO habits (user_id, name, is_sample, created_at) "
                "VALUES (?, ?, 0, ?);",
                (user_id, new_habit, datetime.now().isoformat(timespec="seconds")),
            )
            conn.commit()
            conn.close()
            return redirect(url_for("habits"))

        # if user ticked habits as completed
        completed_ids = request.form.getlist("completed_habits")
        if completed_ids:
            conn = get_db_connection()
            cur = conn.cursor()
            now = datetime.now().isoformat(timespec="seconds")

            for hid in completed_ids:
                try:
                    hid_int = int(hid)
                except ValueError:
                    continue

                # verify habit belongs to this user
                cur.execute(
                    "SELECT id FROM habits WHERE id = ? AND user_id = ?;",
                    (hid_int, user_id),
                )
                if cur.fetchone() is None:
                    continue

                cur.execute(
                    "INSERT INTO habit_completions (habit_id, completed_at) "
                    "VALUES (?, ?);",
                    (hid_int, now),
                )
                # each completed habit counts as identity-aligned behaviour
                update_alignment_score(user_id, 1)

            conn.commit()
            conn.close()
            return redirect(url_for("habits"))

    # show all habits for this user
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, is_sample, created_at "
        "FROM habits WHERE user_id = ? ORDER BY id ASC;",
        (user_id,),
    )
    habits_rows = cur.fetchall()
    conn.close()

    return render_template("habits.html", habits=habits_rows)

@app.route("/analytics")
@login_required
def analytics():
    user_id = session['user_id']
    conn = get_db_connection()
    cur = conn.cursor()

    # 1. emotion distribution across this user's journal entries
    cur.execute("""
        SELECT predicted_emotion, COUNT(*) as count
        FROM journal_entries
        WHERE user_id = ?
        GROUP BY predicted_emotion
    """, (user_id,))
    emotion_data = cur.fetchall()

    # 2. days user journaled in last 7 days
    today = date.today()
    last_week = (today - timedelta(days=6)).isoformat()
    cur.execute("""
        SELECT COUNT(DISTINCT date(created_at)) AS active_days
        FROM journal_entries
        WHERE user_id = ? AND date(created_at) >= ?
    """, (user_id, last_week))
    active_days = cur.fetchone()["active_days"]

    # 3. habit completions in last 7 days
    cur.execute("""
        SELECT COUNT(*) AS habits_done
        FROM habit_completions hc
        JOIN habits h ON hc.habit_id = h.id
        WHERE h.user_id = ? AND date(hc.completed_at) >= ?
    """, (user_id, last_week))
    habits_done = cur.fetchone()["habits_done"]

    conn.close()

    return render_template(
        "analytics.html",
        emotion_data=emotion_data,
        active_days=active_days,
        habits_done=habits_done
    )

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
