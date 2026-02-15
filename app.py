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


def extract_echo_phrase(text: str) -> str:
    """
    Extract a phrase from journal entry to echo back in reframe.
    Priority: trigger phrases > emotionally loaded clauses > first sentence fragment.
    Max 12 words for safety.
    """
    import re
    text = text.strip()

    # Stage 1: Look for specific emotional trigger patterns (verbatim)
    trigger_patterns = [
        # Anxious triggers
        (r"(what if [^.!?]{1,60})", re.IGNORECASE),
        (r"(scared (?:that |it's |I |it )[^.!?]{1,50})", re.IGNORECASE),
        (r"(worried (?:that |about |I )[^.!?]{1,50})", re.IGNORECASE),
        (r"(I (?:open|opened) [^.!?]{1,40})", re.IGNORECASE),
        (r"(I (?:close|closed) [^.!?]{1,40})", re.IGNORECASE),
        (r"(going to be [^.!?]{1,40})", re.IGNORECASE),

        # Overwhelmed triggers
        (r"(too much[^.!?]{0,50})", re.IGNORECASE),
        (r"(don't know where to [^.!?]{1,40})", re.IGNORECASE),
        (r"(everything (?:is |feels )?(?:piling|overwhelming)[^.!?]{0,40})", re.IGNORECASE),
        (r"(so many [^.!?]{1,50})", re.IGNORECASE),

        # Stressed triggers
        (r"(deadline[^.!?]{0,50})", re.IGNORECASE),
        (r"(running out of time[^.!?]{0,40})", re.IGNORECASE),
        (r"(\d+ hours[^.!?]{0,40})", re.IGNORECASE),
        (r"(not enough time[^.!?]{0,40})", re.IGNORECASE),

        # Avoidant triggers
        (r"(I (?:keep |kept )?avoiding[^.!?]{1,50})", re.IGNORECASE),
        (r"(I (?:keep |kept )?putting (?:it |this |off)[^.!?]{1,40})", re.IGNORECASE),
        (r"(ended? up (?:scrolling|watching|doing)[^.!?]{1,40})", re.IGNORECASE),
        (r"(I'll do it (?:later|tomorrow)[^.!?]{0,30})", re.IGNORECASE),

        # Discouraged triggers
        (r"(no point[^.!?]{0,50})", re.IGNORECASE),
        (r"(what's the point[^.!?]{0,40})", re.IGNORECASE),
        (r"(never (?:get|improve|succeed)[^.!?]{1,40})", re.IGNORECASE),
        (r"(always (?:fail|mess)[^.!?]{1,40})", re.IGNORECASE),
    ]

    for pattern, flags in trigger_patterns:
        match = re.search(pattern, text, flags)
        if match:
            phrase = match.group(1).strip()
            # Cap at 12 words for safety
            words = phrase.split()[:12]
            return ' '.join(words).rstrip('.,!?')

    # Stage 2: Extract emotionally loaded sentence (contains emotional keywords)
    emotional_keywords = [
        'overwhelmed', 'anxious', 'scared', 'worried', 'stressed', 'tired',
        'avoiding', 'procrastinating', 'stuck', 'frozen', 'paralyzed',
        'deadline', 'pressure', 'fail', 'terrible', 'hopeless', 'pointless'
    ]

    sentences = re.split(r'[.!?]+', text)
    for sentence in sentences:
        sentence = sentence.strip()
        if any(kw in sentence.lower() for kw in emotional_keywords):
            # Extract first clause (up to comma or 10 words)
            clause = re.split(r',', sentence)[0].strip()
            words = clause.split()[:10]
            if len(words) >= 3:
                return ' '.join(words).rstrip('.,!?')

    # Stage 3: Fallback - first 8 words
    words = text.split()[:8]
    return ' '.join(words).rstrip('.,!?')


def select_template_index(text_lower: str, emotion: str) -> int:
    """
    Deterministically select template index based on text features.
    Returns 0-based index for template array.
    """
    # Anxious: 5 templates
    if emotion == "anxious":
        if "what if" in text_lower:
            return 0  # worry loop template
        elif "open" in text_lower and "close" in text_lower:
            return 1  # avoidance cycle template
        elif "terrible" in text_lower or "awful" in text_lower or "bad" in text_lower:
            return 2  # catastrophizing template
        elif "everyone" in text_lower or "people" in text_lower or "judge" in text_lower:
            return 3  # social fear template
        else:
            return 4  # general anxious template

    # Overwhelmed: 4 templates
    elif emotion == "overwhelmed":
        if "don't know where" in text_lower or "where to start" in text_lower:
            return 0  # paralysis from choice template
        elif "too much" in text_lower or "so many" in text_lower:
            return 1  # volume overload template
        elif "everything" in text_lower and ("piling" in text_lower or "at once" in text_lower):
            return 2  # holding all at once template
        else:
            return 3  # general overwhelmed template

    # Stressed: 4 templates
    elif emotion == "stressed":
        if "deadline" in text_lower or "hours" in text_lower or "due" in text_lower:
            return 0  # time urgency template
        elif "pressure" in text_lower or "rushing" in text_lower:
            return 1  # pressure reducing thinking template
        elif "can't focus" in text_lower or "can't think" in text_lower:
            return 2  # stress blocking cognition template
        else:
            return 3  # general stressed template

    # Avoidant: 5 templates
    elif emotion == "avoidant":
        if ("open" in text_lower and "close" in text_lower) or ("start" in text_lower and "stop" in text_lower):
            return 0  # start-stop cycle template
        elif "avoiding" in text_lower or "putting it off" in text_lower:
            return 1  # explicit avoidance template
        elif "later" in text_lower or "tomorrow" in text_lower:
            return 2  # postponing template
        elif "ended up" in text_lower or "instead" in text_lower:
            return 3  # distraction template
        else:
            return 4  # general avoidant template

    # Discouraged: 4 templates
    elif emotion == "discouraged":
        if "no point" in text_lower or "what's the point" in text_lower:
            return 0  # hopelessness template
        elif "never" in text_lower or "always" in text_lower:
            return 1  # absolutist thinking template
        elif "not good enough" in text_lower or "can't do" in text_lower:
            return 2  # low self-belief template
        else:
            return 3  # general discouraged template

    # Calm: 2 templates
    elif emotion == "calm":
        if "clear" in text_lower or "focused" in text_lower:
            return 0  # clarity template
        else:
            return 1  # general calm template

    # Neutral/other: 1 template
    else:
        return 0


def generate_reframe(journal_text: str, emotion_label: str) -> str:
    """
    Generate emotionally intelligent reframe that:
    1) Echoes a phrase from journal_text
    2) Identifies perceived threat (plain language)
    3) Shifts interpretation (not just reassurance)
    4) Restores agency with ≤2min concrete action

    Returns: 2-4 sentence reframe string
    """
    text_lower = journal_text.lower()
    echo = extract_echo_phrase(journal_text)

    # Map 'stuck' to avoidant for compatibility
    if emotion_label == "stuck":
        emotion_label = "avoidant"

    template_idx = select_template_index(text_lower, emotion_label)

    # === ANXIOUS TEMPLATES ===
    if emotion_label == "anxious":
        templates = [
            # 0: worry loop
            f'When you ask "{echo}", your brain is running disaster simulations to try to prevent the worst — but the question doesn\'t have an answer right now. It\'s anxiety looking for something to grab onto. Write down one tiny thing you can control in the next 10 minutes, even if it doesn\'t solve the whole worry.',

            # 1: avoidance cycle (open/close)
            f'The pattern you describe — "{echo}" — is your brain treating *starting* as the threat, not the work itself. The fear spikes at the threshold, not during the task. Set a 2-minute timer and agree to just sit with the open document. You don\'t have to write; just practice staying.',

            # 2: catastrophizing
            f'The fear that "{echo}" isn\'t a preview of what will happen — it\'s your nervous system amplifying the threat to keep you safe. But avoidance makes the fear stronger. Write one ugly sentence with zero expectation it\'ll be good. Just prove the task won\'t destroy you.',

            # 3: social fear
            f'You\'re comparing your internal panic to other people\'s external calm, which makes the gap feel impossible. But their composure doesn\'t mean you\'re failing — you just can\'t see their nerves. Do one small visible action (open notes, write one sentence) to anchor yourself in what you can control, not what they might think.',

            # 4: general anxious
            f'Worry feels productive because it keeps your brain occupied, but it\'s not the same as preparation. The outcome you\'re afraid of isn\'t decided by how much you worry right now. Pick the smallest step you can take in the next 5 minutes — even reading one paragraph — and let that be enough for this moment.',
        ]
        return templates[template_idx]

    # === OVERWHELMED TEMPLATES ===
    elif emotion_label == "overwhelmed":
        templates = [
            # 0: paralysis from choice
            f'When you say "{echo}", you\'re stuck because your brain thinks there\'s a "right" first choice. But there isn\'t — any task breaks the paralysis. Pick the one with the nearest deadline or the one that takes the least energy. Either answer works.',

            # 1: volume overload
            f'Your brain is trying to hold all of "{echo}" at the same time, which creates cognitive overload. You don\'t need to solve everything right now. Choose one task — just one — and do it for 10 minutes. Everything else can wait outside your attention.',

            # 2: holding all at once
            f'You\'re seeing all your tasks as one enormous thing instead of separate small pieces. The feeling of "{echo}" is your brain collapsing under the weight of everything simultaneously. Write down three tasks, then hide the list and do only the top one for the next 15 minutes.',

            # 3: general overwhelmed
            f'Overwhelm is what happens when your brain refuses to prioritize because everything feels urgent. But urgency is a feeling, not a fact. Pick the task that would give you the most relief if it were done, or the smallest one. Do that first, even badly.',
        ]
        return templates[template_idx]

    # === STRESSED TEMPLATES ===
    elif emotion_label == "stressed":
        templates = [
            # 0: time urgency
            f'The time pressure you feel — "{echo}" — is making everything seem impossible, but you\'ve already survived this long under the same conditions. Your brain doesn\'t need calm or clarity right now; it just needs you to move your hands for the next 30 minutes and see what emerges.',

            # 1: pressure reducing thinking
            f'When you say "{echo}", you\'re noticing how pressure narrows your thinking and makes focus harder. That\'s your stress response, not a reflection of what\'s actually possible. Set a 20-minute timer and commit to messy, imperfect output. Done badly is better than not done.',

            # 2: stress blocking cognition
            f'Stress is blocking your ability to think clearly, which makes you feel like you can\'t do the work — but that\'s the stress, not the task. You don\'t need mental clarity to make progress; you need motion. Open the document and type anything for 10 minutes, even if it\'s nonsense. Movement creates clarity, not the other way around.',

            # 3: general stressed
            f'The urgency makes your brain think everything has to be perfect immediately, but perfection isn\'t possible under time pressure. Messy progress still counts. Do the next most obvious thing for 15 minutes without judging the quality. You can fix it later; right now you just need words on the page.',
        ]
        return templates[template_idx]

    # === AVOIDANT TEMPLATES ===
    elif emotion_label == "avoidant":
        templates = [
            # 0: start-stop cycle
            f'The cycle you describe — "{echo}" — shows your brain treating the task like a threat every time you approach it. But avoidance is a relief loop: you get short-term comfort, which trains your brain to avoid harder next time. Break the loop: open the file and leave it open for 5 minutes without doing anything. Just practice not closing it.',

            # 1: explicit avoidance
            f'When you say "{echo}", you\'re noticing the gap between knowing what to do and actually doing it. That gap isn\'t laziness — it\'s your brain seeking relief from discomfort. But the relief is temporary and the task grows scarier. Do one micro-step (write one sentence, read one paragraph) without expecting yourself to finish. Just prove you can start.',

            # 2: postponing
            f'The thought "{echo}" is your brain bargaining for relief right now by promising future action. But "later" never feels better than now — it just moves the discomfort. You don\'t need to feel ready or motivated. Set a 2-minute timer and do the absolute smallest version of the task. Two minutes. That\'s it.',

            # 3: distraction
            f'You noticed that "{echo}" — that\'s the relief loop in action. The distraction feels better than the discomfort of starting, so your brain learns to reach for it every time. But the task doesn\'t go away; it just gets harder. Do one tiny thing (open one source, write one bullet point) before you allow the next distraction. Make starting the price of scrolling.',

            # 4: general avoidant
            f'You already know what needs to be done — that\'s not the problem. The problem is your brain is waiting for a "ready" feeling that never comes. You don\'t need to feel motivated or prepared. Start badly, start messy, start afraid. Just start. Write one imperfect sentence and see what happens.',
        ]
        return templates[template_idx]

    # === DISCOURAGED TEMPLATES ===
    elif emotion_label == "discouraged":
        templates = [
            # 0: hopelessness
            f'When you say "{echo}", you\'re measuring the value of trying by outcomes you can\'t control yet. But trying isn\'t pointless just because success isn\'t guaranteed. Showing up is the point. Making an attempt is the point. Do one small thing today — even 5 minutes of work — to prove to yourself that effort still counts, even when the outcome is uncertain.',

            # 1: absolutist thinking
            f'You\'re using words like "{echo}" to describe your pattern, but absolutes collapse nuance. You\'re measuring yourself by isolated moments instead of across time. Progress isn\'t linear — sometimes you only see it when you look backward. Do one thing differently today, even tiny, to break the "always/never" story your brain is telling.',

            # 2: low self-belief
            f'The belief that "{echo}" is measuring your identity by a single outcome or task. But one result doesn\'t define your capacity. The gap between where you are and where you want to be exists *because* you\'re learning what "better" looks like — that\'s growth, not failure. Prove you can try by doing one small thing imperfectly today.',

            # 3: general discouraged
            f'You\'re seeing the difficulty as evidence you\'re not capable, but struggle is part of learning, not proof of inadequacy. The fact that it\'s hard doesn\'t mean it\'s impossible. Do the smallest possible version of the task — one sentence, one problem, one paragraph — just to prove movement is still possible, even when belief is low.',
        ]
        return templates[template_idx]

    # === CALM TEMPLATES ===
    elif emotion_label == "calm":
        templates = [
            # 0: clarity
            f'This clarity you\'re feeling — where "{echo}" — is worth noticing and protecting. You don\'t need to rush to capitalize on it or do everything at once. Just take one gentle next step while the calm is here, and let that be enough.',

            # 1: general calm
            f'The steadiness you\'re experiencing right now is valuable. You don\'t need to amplify it or make it productive immediately. Just notice it, and do one small thing that aligns with how you want to feel. Let the calm guide the action, not the urgency.',
        ]
        return templates[template_idx]

    # === NEUTRAL / OTHER ===
    else:
        return f'You showed up and wrote this down — that\'s already movement, even if it doesn\'t feel significant. You don\'t need to have it all figured out to take one small next step. Pick the easiest or nearest task and do it for 10 minutes, without judgment.'


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
