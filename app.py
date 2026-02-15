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


def extract_key_phrase(text: str) -> str:
    """Extract a meaningful phrase from journal entry (up to 8 words)"""
    import re
    text = text.strip()

    # Look for meaningful phrases with emotional keywords
    patterns = [
        r"(too much\b[^.!?]{0,40})",
        r"(don't know where to\b[^.!?]{0,40})",
        r"(what if\b[^.!?]{0,40})",
        r"(scared that\b[^.!?]{0,40})",
        r"(worried\b[^.!?]{0,40})",
        r"(deadline\b[^.!?]{0,40})",
        r"(running out of time\b[^.!?]{0,40})",
        r"(keep putting\b[^.!?]{0,40})",
        r"(avoiding\b[^.!?]{0,40})",
        r"(ended up\b[^.!?]{0,40})",
        r"(no point\b[^.!?]{0,40})",
        r"(never\b[^.!?]{0,40})",
        r"(can't\b[^.!?]{0,30})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            phrase = match.group(1).strip()
            words = phrase.split()[:8]
            return ' '.join(words)

    # Fallback: first 6-8 words
    words = text.split()[:7]
    return ' '.join(words)


def generate_reframe(journal_text: str, emotion: str) -> str:
    """
    Generate emotionally intelligent reframe based on journal text and emotion.
    Echoes user's words and provides specific perspective shift.
    """
    text_lower = journal_text.lower()
    key_phrase = extract_key_phrase(journal_text)

    # Map 'stuck' to avoidant-style reframes (current model doesn't have 'avoidant')
    if emotion == "stuck":
        emotion = "avoidant"

    # Emotion-specific reframe patterns
    if emotion == "overwhelmed":
        if "don't know where" in text_lower or "where to start" in text_lower:
            return f"The fact that you can list out what needs doing means it's not actually unknowable. Your brain just needs permission to do one thing at a time instead of holding all of it at once."
        elif "too much" in text_lower or "piling up" in text_lower:
            return f"When you say '{key_phrase}', your brain is trying to hold everything at once. Pick the smallest task or the nearest deadline — either answer breaks the freeze."
        else:
            return "You're seeing all the tasks as one giant thing instead of separate small actions. Your brain doesn't need to solve everything right now — just the next single step."

    elif emotion == "anxious":
        if "what if" in text_lower:
            return f"Your brain is running disaster simulations on repeat because it thinks preparing for the worst will prevent it. But '{key_phrase}' isn't a question that has an answer right now — it's just anxiety looking for something to hold onto."
        elif "fail" in text_lower or "mess up" in text_lower or "scared" in text_lower:
            return f"The fear you're feeling isn't a preview of what will happen — it's just your nervous system treating the future like it's already here. You can't control the outcome, but you can control what you do in the next ten minutes."
        elif "everyone else" in text_lower or "prepared" in text_lower:
            return "Comparing your internal panic to everyone else's external appearance makes the gap feel bigger than it is. Their calm doesn't mean you're failing — it just means you can't see their nerves."
        else:
            return "Worry feels productive because it keeps your brain busy, but it's not preparation — it's just loops. You don't need certainty to take one small action."

    elif emotion == "stressed":
        if "deadline" in text_lower or "hours" in text_lower or "running out" in text_lower:
            return f"The time pressure is making everything feel impossible, but you've already done something under the worst conditions. Your brain doesn't need clarity right now — it just needs you to keep your hands moving for the next hour."
        elif "pressure" in text_lower or "can't focus" in text_lower:
            return "The racing feeling is your body's stress response, not a reflection of what's actually possible. You don't need to feel calm to make progress — you just need to move your hands while your heart races."
        else:
            return "Urgency makes your brain think everything has to be perfect immediately. But messy progress made under pressure still counts as progress."

    elif emotion == "avoidant":
        if "later" in text_lower or "tomorrow" in text_lower or "maybe" in text_lower:
            return f"Mental tiredness often shows up right when you're about to do something that requires thinking, then mysteriously lifts when you're doing something easier. You might not need rest — you might just need to open the file for two minutes without expecting yourself to finish."
        elif "avoiding" in text_lower or "putting it off" in text_lower or "three days" in text_lower:
            return f"Avoidance tells you the task feels like something you need to be 'ready' for. You don't need to do it well or even finish it. You just need to do one small piece badly and see what happens next."
        elif "ended up" in text_lower or "instead" in text_lower:
            return "The phone and the distractions aren't signs you can't do this — they're just your brain's way of avoiding discomfort. You don't need to feel motivated to begin; you just need to start before you feel ready."
        else:
            return "You already know what to do — that's not the problem. The problem is your brain is waiting for the 'right' feeling that never comes. Start messy. Start badly. Just start."

    elif emotion == "discouraged":
        if "no point" in text_lower or "what's the point" in text_lower:
            return f"When you say '{key_phrase}', you're measuring success by outcomes you can't control yet. Showing up is the point. Trying is the point. Those count even when the results aren't visible yet."
        elif "never" in text_lower or "always" in text_lower:
            return "You're measuring improvement by looking at single moments in isolation instead of across time. Progress isn't linear — sometimes you only see it when you look backward, not when you're in the middle."
        else:
            return "The gap between where you are and where you want to be only exists because you're learning what 'better' looks like. That gap is proof you're growing, not proof you're failing."

    elif emotion == "calm":
        return "This clarity you're feeling right now is worth noticing. You don't need to rush to capitalize on it — just take the next small step while it's here."

    else:
        # neutral or unrecognized
        return "You showed up and wrote this down — that's already movement. You don't need to have it all figured out to take one small next step."


def get_affirmation(predicted_emotion: str) -> str:
    """Return identity-based affirmation for given emotion"""
    affirmations = {
        "overwhelmed": "I can take this slowly and still make progress.",
        "anxious": "I can steady myself and begin gently.",
        "stuck": "I move forward in small steps even when it feels hard.",
        "stressed": "I can pace myself and still get things done.",
        "tired": "I am allowed to pause without losing progress.",
        "calm": "I am becoming the version of myself I imagined.",
    }
    return affirmations.get(predicted_emotion, "I am doing my best and that is enough.")


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
        today_affirmation = get_affirmation(emotion)

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
            reframe = generate_reframe(entry_text, predicted_emotion)
            affirmation = get_affirmation(predicted_emotion)

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
