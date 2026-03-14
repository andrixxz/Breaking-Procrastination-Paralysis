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

# model file paths — emotion classifier (Model 1)
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.pkl")

# model file paths — behaviour state classifier (Model 2)
BEHAVIOUR_VECTORIZER_PATH = os.path.join(MODEL_DIR, "behaviour_vectorizer.pkl")
BEHAVIOUR_MODEL_PATH = os.path.join(MODEL_DIR, "behaviour_model.pkl")

# load ML models — emotion classifier (always required)
vectorizer = joblib.load(VECTORIZER_PATH)
emotion_model = joblib.load(MODEL_PATH)

# load ML models — behaviour state classifier (graceful fallback if not yet trained)
behaviour_vectorizer = None
behaviour_model = None
if os.path.exists(BEHAVIOUR_VECTORIZER_PATH) and os.path.exists(BEHAVIOUR_MODEL_PATH):
    behaviour_vectorizer = joblib.load(BEHAVIOUR_VECTORIZER_PATH)
    behaviour_model = joblib.load(BEHAVIOUR_MODEL_PATH)
    print("Behaviour state classifier loaded successfully.")
else:
    print("Warning: Behaviour model files not found. Run train_model.py to generate them.")


# database setup + connection

def get_db_connection():
    # opens DB with rows returned as dictionary-like objects
    conn = sqlite3.connect(DB_PATH, timeout=30)  # Increased timeout to 30 seconds
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Enable Write-Ahead Logging for better concurrency
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
            created_at TEXT NOT NULL,
            onboarding_complete INTEGER NOT NULL DEFAULT 0
        );
    """)

    # journal entries table (per user)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS journal_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            entry_text TEXT NOT NULL,
            predicted_emotion TEXT NOT NULL,
            predicted_behaviour TEXT,
            paralysis_score REAL,
            reframe TEXT,
            micro_task_text TEXT,
            micro_task_minutes INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)

    # todos table (per user)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS todos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'manual',
            journal_entry_id INTEGER,
            due_date TEXT,
            is_done INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (journal_entry_id) REFERENCES journal_entries(id)
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


def migrate_db_for_onboarding():
    """Add onboarding tables and update existing schema"""
    conn = get_db_connection()
    cur = conn.cursor()

    # 1. Add onboarding_complete to users table (if not exists)
    cur.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cur.fetchall()]
    if 'onboarding_complete' not in columns:
        cur.execute("ALTER TABLE users ADD COLUMN onboarding_complete INTEGER NOT NULL DEFAULT 0")

    # 2. Create goals table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            goal_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)

    # 3. Create identity_beliefs table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS identity_beliefs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            belief_text TEXT NOT NULL,
            linked_goal_id INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (linked_goal_id) REFERENCES goals(id)
        );
    """)

    # 4. Create positive_thoughts table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS positive_thoughts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            thought_text TEXT NOT NULL,
            linked_belief_id INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (linked_belief_id) REFERENCES identity_beliefs(id)
        );
    """)

    # 5. Add linked_goal_id to habits table (if not exists)
    cur.execute("PRAGMA table_info(habits)")
    habit_columns = [col[1] for col in cur.fetchall()]
    if 'linked_goal_id' not in habit_columns:
        cur.execute("ALTER TABLE habits ADD COLUMN linked_goal_id INTEGER")

    # 6. Add reframe + micro-task columns to journal_entries table (if not exists)
    cur.execute("PRAGMA table_info(journal_entries)")
    journal_columns = [col[1] for col in cur.fetchall()]
    if 'reframe' not in journal_columns:
        cur.execute("ALTER TABLE journal_entries ADD COLUMN reframe TEXT")
    if 'micro_task_text' not in journal_columns:
        cur.execute("ALTER TABLE journal_entries ADD COLUMN micro_task_text TEXT")
    if 'micro_task_minutes' not in journal_columns:
        cur.execute("ALTER TABLE journal_entries ADD COLUMN micro_task_minutes INTEGER")
    if 'predicted_behaviour' not in journal_columns:
        cur.execute("ALTER TABLE journal_entries ADD COLUMN predicted_behaviour TEXT")
    if 'paralysis_score' not in journal_columns:
        cur.execute("ALTER TABLE journal_entries ADD COLUMN paralysis_score REAL")

    # 7. Create todos table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS todos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'manual',
            journal_entry_id INTEGER,
            due_date TEXT,
            is_done INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (journal_entry_id) REFERENCES journal_entries(id)
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


def onboarding_check(f):
    """Redirect to onboarding if not completed"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('landing'))

        # Check if onboarding is complete
        user_id = session['user_id']
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT onboarding_complete FROM users WHERE id = ?", (user_id,))
            user = cur.fetchone()

            if user and user['onboarding_complete'] == 0:
                # Determine which onboarding step to redirect to
                return redirect(get_next_onboarding_step(user_id))
        finally:
            conn.close()

        return f(*args, **kwargs)
    return decorated_function


def get_next_onboarding_step(user_id):
    """Determine which onboarding step user should complete next"""
    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Check if goals exist
        cur.execute("SELECT COUNT(*) as count FROM goals WHERE user_id = ?", (user_id,))
        goals_count = cur.fetchone()['count']
        if goals_count == 0:
            return url_for('onboarding_goals')

        # Check if beliefs exist
        cur.execute("SELECT COUNT(*) as count FROM identity_beliefs WHERE user_id = ?", (user_id,))
        beliefs_count = cur.fetchone()['count']
        if beliefs_count == 0:
            return url_for('onboarding_beliefs')

        # Check if thoughts exist
        cur.execute("SELECT COUNT(*) as count FROM positive_thoughts WHERE user_id = ?", (user_id,))
        thoughts_count = cur.fetchone()['count']
        if thoughts_count == 0:
            return url_for('onboarding_thoughts')

        # Check if custom habits exist (non-sample)
        cur.execute("SELECT COUNT(*) as count FROM habits WHERE user_id = ? AND is_sample = 0", (user_id,))
        habits_count = cur.fetchone()['count']

        if habits_count == 0:
            return url_for('onboarding_habits')

        # All steps complete - shouldn't reach here
        return url_for('today')
    finally:
        conn.close()


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
    """Predict emotion from journal text using Model 1."""
    X = vectorizer.transform([text])
    pred = emotion_model.predict(X)[0]
    return pred


def predict_behaviour(text: str) -> str:
    """Predict behaviour state from journal text using Model 2. Returns None if model not loaded."""
    if behaviour_model is None or behaviour_vectorizer is None:
        return None
    try:
        X = behaviour_vectorizer.transform([text])
        pred = behaviour_model.predict(X)[0]
        return pred
    except Exception:
        return None


# -- Paralysis Score Algorithm (Task 2.4) --
# Combines behaviour state, emotion, keyword signals and daily frequency
# into a single score that quantifies how frozen the user is.
# Negative = productive/flowing, Positive = paralysed/stuck.

# Behaviour state contribution to paralysis score
BEHAVIOUR_WEIGHTS = {
    "avoidance": 3,
    "overwhelm": 2,
    "rumination": 1,
    "recovery": -1,
    "action": -2,
    "completion": -3,
}

# Emotion contribution to paralysis score
EMOTION_WEIGHTS = {
    "guilty": 2,
    "anxious": 2,
    "overwhelmed": 2,
    "stressed": 1,
    "unmotivated": 1,
    "frustrated": 1,
    "stuck": 1,
    "tired": 0,
    "calm": -1,
    "hopeful": -2,
    "proud": -2,
}

# Words that signal deeper paralysis when present in journal text
PARALYSIS_KEYWORDS = ["can't", "never", "always", "hate", "impossible"]


def calculate_paralysis_score(emotion, behaviour, entry_text, user_id):
    """Calculate paralysis score from emotion, behaviour, keywords, and daily frequency."""

    # Edge case: no text provided
    if not entry_text or not entry_text.strip():
        return None

    # Edge case: emotion is required at minimum
    if not emotion:
        return None

    raw_score = 0.0

    # 1. Behaviour state weight (skip if model unavailable)
    if behaviour and behaviour in BEHAVIOUR_WEIGHTS:
        raw_score += BEHAVIOUR_WEIGHTS[behaviour]

    # 2. Emotion weight
    if emotion in EMOTION_WEIGHTS:
        raw_score += EMOTION_WEIGHTS[emotion]
    # Edge case: unknown emotion label - treat as neutral (0)

    # 3. Keyword boost - presence of paralysis-signalling words
    # +1 per keyword found, capped at +3
    text_lower = entry_text.lower()
    keyword_count = 0
    for keyword in PARALYSIS_KEYWORDS:
        if keyword in text_lower:
            keyword_count += 1
    keyword_boost = min(keyword_count, 3)
    raw_score += keyword_boost

    # 4. Frequency factor - 3rd+ negative entry today means stuck in a loop
    # Negative states are emotions with weight >= 1 or behaviours of avoidance/overwhelm/rumination
    negative_emotions = [e for e, w in EMOTION_WEIGHTS.items() if w >= 1]
    try:
        today_str = date.today().isoformat()
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT predicted_emotion FROM journal_entries "
                "WHERE user_id = ? AND created_at LIKE ?",
                (user_id, today_str + "%"),
            )
            today_entries = cur.fetchall()
        finally:
            conn.close()

        # Count how many of today's previous entries had negative emotions
        negative_count = sum(
            1 for row in today_entries
            if row["predicted_emotion"] in negative_emotions
        )
        # If this new entry is also negative, add 1 to count for the check
        if emotion in negative_emotions:
            negative_count += 1
        # 3rd or more negative entry today adds +1
        if negative_count >= 3:
            raw_score += 1
    except Exception:
        # Edge case: DB error during frequency check - skip this factor
        pass

    # 5. Normalise to -5 to +5 range
    # Theoretical raw range: behaviour(-3 to +3) + emotion(-2 to +2) + keywords(0 to +3) + frequency(0 to +1) = -5 to +9
    # Without behaviour model: emotion(-2 to +2) + keywords(0 to +3) + frequency(0 to +1) = -2 to +6
    # Clamp to -5 to +5
    score = max(-5.0, min(5.0, raw_score))

    # Round to one decimal place for clean storage and display
    return round(score, 1)


def get_paralysis_label(score):
    """Return a human-readable label and CSS class for the paralysis score."""
    if score is None:
        return None, None
    if score <= -3:
        return "In flow", "flow"
    elif score <= -1:
        return "Moving forward", "forward"
    elif score <= 1:
        return "Neutral", "neutral"
    elif score <= 3:
        return "Some resistance", "resistance"
    else:
        return "High paralysis", "paralysis"


# -- Temporal Analysis (Task 2.5) --
# Detects emotion and behaviour transitions across same-day journal entries
# and generates a personalised insight. Aligns with supervisor feedback:
# "Journal entries same day should relate to each other."

# Emotions grouped by valence for transition detection
POSITIVE_EMOTIONS = {"calm", "hopeful", "proud"}
NEGATIVE_EMOTIONS = {"overwhelmed", "anxious", "stuck", "stressed",
                     "frustrated", "guilty", "unmotivated"}
# "tired" is neutral - not in either group

# Behaviour states grouped for transition detection
POSITIVE_BEHAVIOURS = {"action", "completion", "recovery"}
NEGATIVE_BEHAVIOURS = {"avoidance", "overwhelm", "rumination"}

# -- Belief bridge templates for three-layer reframes (Task 2.7) --
# Each bridge connects the user's identity belief to their current
# behaviour state, creating a personalised closing sentence in the reframe.
BELIEF_BRIDGES_BY_BEHAVIOUR = {
    "avoidance": "Taking this small step is one way to live that out.",
    "overwhelm": "Choosing just one thing right now is how that starts.",
    "rumination": "Moving from thinking to doing, even briefly, is how that becomes real.",
    "action": "What you are doing right now is proof of that.",
    "completion": "What you just finished is evidence of that.",
    "recovery": "Resting so you can return stronger is part of that.",
}

# Emotion-only bridges used when behaviour state is unavailable
BELIEF_BRIDGES_BY_EMOTION = {
    "calm": "This steadiness is part of that.",
    "hopeful": "This feeling is evidence of that.",
    "proud": "What you accomplished is proof of that.",
    "overwhelmed": "Choosing one small thing right now is how that starts.",
    "anxious": "One small action from here is how that becomes real.",
    "stuck": "The next small step is how that becomes real.",
    "stressed": "One rough action right now is how that starts.",
    "tired": "Showing up even now is part of that.",
    "frustrated": "Channeling this energy into one action is how that becomes real.",
    "guilty": "One forward step right now is how that starts again.",
    "unmotivated": "Acting without motivation is the strongest form of that.",
}


def get_daily_insight(user_id):
    """Analyse today's journal entries and return a temporal insight message."""

    today_str = date.today().isoformat()

    try:
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT predicted_emotion, predicted_behaviour, paralysis_score "
                "FROM journal_entries "
                "WHERE user_id = ? AND created_at LIKE ? "
                "ORDER BY id ASC",
                (user_id, today_str + "%"),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
    except Exception:
        # DB error - return no insight rather than crashing
        return None

    # Need at least 2 entries to detect a transition
    if len(rows) < 2:
        return None

    emotions = [row["predicted_emotion"] for row in rows if row["predicted_emotion"]]
    behaviours = [row["predicted_behaviour"] for row in rows if row["predicted_behaviour"]]
    scores = [row["paralysis_score"] for row in rows if row["paralysis_score"] is not None]

    # Edge case: all emotions are None (should not happen, but safe to check)
    if len(emotions) < 2:
        return None

    first_emotion = emotions[0]
    last_emotion = emotions[-1]
    entry_count = len(rows)

    # Priority 1: Detect negative loop (same negative emotion 3+ times)
    if entry_count >= 3:
        # Check if the most common emotion appears 3+ times and is negative
        emotion_counts = {}
        for e in emotions:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1

        most_common = max(emotion_counts, key=emotion_counts.get)
        if emotion_counts[most_common] >= 3 and most_common in NEGATIVE_EMOTIONS:
            return {
                "type": "negative_loop",
                "message": (
                    f"You have been sitting with feeling {most_common} today. "
                    "That is okay. Noticing it is the first step."
                ),
                "css_class": "insight-loop",
            }

    # Priority 2: Detect behaviour loop (same negative behaviour 3+ times)
    if len(behaviours) >= 3:
        behaviour_counts = {}
        for b in behaviours:
            behaviour_counts[b] = behaviour_counts.get(b, 0) + 1

        most_common_beh = max(behaviour_counts, key=behaviour_counts.get)
        if behaviour_counts[most_common_beh] >= 3 and most_common_beh in NEGATIVE_BEHAVIOURS:
            return {
                "type": "behaviour_loop",
                "message": (
                    f"You have been in a pattern of {most_common_beh} today. "
                    "That is okay. Awareness is what breaks the cycle."
                ),
                "css_class": "insight-loop",
            }

    # Priority 3: Positive transition (negative -> positive emotion)
    if first_emotion in NEGATIVE_EMOTIONS and last_emotion in POSITIVE_EMOTIONS:
        return {
            "type": "positive_transition",
            "message": (
                f"Today you moved from feeling {first_emotion} to feeling "
                f"{last_emotion}. That is your brain building a new pathway."
            ),
            "css_class": "insight-positive",
        }

    # Priority 4: Behaviour shift (negative -> positive behaviour)
    if len(behaviours) >= 2:
        first_beh = behaviours[0]
        last_beh = behaviours[-1]
        if first_beh in NEGATIVE_BEHAVIOURS and last_beh in POSITIVE_BEHAVIOURS:
            return {
                "type": "behaviour_shift",
                "message": (
                    f"Today you shifted from {first_beh} to {last_beh}. "
                    "That shift took courage. Your brain noticed."
                ),
                "css_class": "insight-positive",
            }

    # Priority 5: Paralysis score improvement (first score > last score, meaningful drop)
    if len(scores) >= 2:
        first_score = scores[0]
        last_score = scores[-1]
        # A drop of 2+ points is a meaningful improvement
        if first_score - last_score >= 2:
            return {
                "type": "score_improvement",
                "message": (
                    "Your paralysis score dropped across today's entries. "
                    "Each entry helped you process a little more."
                ),
                "css_class": "insight-positive",
            }

    # Priority 6: Emotion changed but not clearly positive or negative
    if first_emotion != last_emotion:
        return {
            "type": "emotion_change",
            "message": (
                f"Today you moved from feeling {first_emotion} to feeling "
                f"{last_emotion}. Emotions shift. Naming them is what matters."
            ),
            "css_class": "insight-neutral",
        }

    # Priority 7: Multiple entries, same emotion throughout (not negative loop since < 3)
    if entry_count == 2 and first_emotion == last_emotion:
        if first_emotion in POSITIVE_EMOTIONS:
            return {
                "type": "positive_consistency",
                "message": (
                    f"You journaled twice today and stayed in a {first_emotion} "
                    "space both times. That steadiness is worth noticing."
                ),
                "css_class": "insight-positive",
            }
        else:
            return {
                "type": "processing",
                "message": (
                    f"You showed up twice today while feeling {first_emotion}. "
                    "That takes something. You are processing, and that counts."
                ),
                "css_class": "insight-neutral",
            }

    # Fallback: multiple entries, general acknowledgment
    return {
        "type": "general",
        "message": (
            f"You journaled {entry_count} times today. Each entry helped you "
            "understand yourself a little better."
        ),
        "css_class": "insight-neutral",
    }


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

    # Tired: 3 templates
    elif emotion == "tired":
        if "sleep" in text_lower or "slept" in text_lower or "insomnia" in text_lower:
            return 0  # sleep deprivation template
        elif "burnout" in text_lower or "burnt" in text_lower or "weeks" in text_lower:
            return 1  # burnout template
        else:
            return 2  # general tired template

    # Frustrated: 4 templates
    elif emotion == "frustrated":
        if "myself" in text_lower or "angry at me" in text_lower:
            return 0  # self-directed anger template
        elif "broken" in text_lower or "not working" in text_lower or "keeps" in text_lower:
            return 1  # system frustration template
        elif "again" in text_lower or "same mistake" in text_lower:
            return 2  # repeated failure template
        else:
            return 3  # general frustrated template

    # Guilty: 4 templates
    elif emotion == "guilty":
        if "let" in text_lower and ("down" in text_lower or "people" in text_lower):
            return 0  # letting others down template
        elif "wasted" in text_lower or "spent" in text_lower:
            return 1  # wasting time template
        elif "should have" in text_lower or "could have" in text_lower:
            return 2  # should-have template
        else:
            return 3  # general guilty template

    # Unmotivated: 3 templates
    elif emotion == "unmotivated":
        if "point" in text_lower or "why" in text_lower or "matter" in text_lower:
            return 0  # existential apathy template
        elif "used to" in text_lower or "anymore" in text_lower:
            return 1  # lost interest template
        else:
            return 2  # general unmotivated template

    # Hopeful: 2 templates
    elif emotion == "hopeful":
        if "progress" in text_lower or "managed" in text_lower or "started" in text_lower:
            return 0  # evidence-based hope template
        else:
            return 1  # general hopeful template

    # Proud: 2 templates
    elif emotion == "proud":
        if "finished" in text_lower or "completed" in text_lower or "submitted" in text_lower:
            return 0  # completion pride template
        else:
            return 1  # general proud template

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

    # Map internal labels for reframe template compatibility
    if emotion_label == "stuck":
        emotion_label = "avoidant"
    elif emotion_label == "unmotivated":
        emotion_label = "discouraged"

    template_idx = select_template_index(text_lower, emotion_label)

    # === ANXIOUS TEMPLATES ===
    if emotion_label == "anxious":
        templates = [
            # 0: worry loop
            f'The question "{echo}" is a trap. It demands certainty about something unknowable, which keeps you spinning without moving. Anxiety mistakes repetition for preparation. What you need is not an answer to the question. Stop asking it. Pick one action that exists outside the loop: send a message, open a file, write one sentence, and watch the urgency lose its grip.',

            # 1: avoidance cycle (open/close)
            f'"{echo}" is threshold fear. The moment before starting feels unbearable, so you close the distance and retreat. But the fear isn\'t about the work; it\'s about crossing from safe to uncertain. The work itself is neutral. Open the file and leave it visible for 5 minutes without touching it. Let proximity replace avoidance as the new normal.',

            # 2: catastrophizing
            f'The terror that "{echo}" is your mind confusing *feeling* in danger with *being* in danger. Fear amplifies the stakes to justify its own intensity, but the intensity isn\'t evidence. The task won\'t collapse you. Write one rough sentence with the explicit goal of it being bad. Show yourself the gap between dread and reality.',

            # 3: social fear
            f'You\'re trapped between your inner experience and what you imagine others see, and that gap feels like proof something is wrong. But you\'re comparing your fear to their façade. They can\'t see your panic, and you can\'t see their doubt. Do one external action (open your notes, type one line) to anchor yourself in what\'s real, not what you think they think.',

            # 4: general anxious
            f'Worry is rehearsal disguised as protection, it creates the illusion of control over outcomes you can\'t influence yet. But spinning doesn\'t prevent disaster; it just exhausts you before anything happens. The way out isn\'t calm; it\'s movement. Pick the smallest concrete step available right now and take it. Let action replace rumination as the thing you do when uncertain.',
        ]
        return templates[template_idx]

    # === OVERWHELMED TEMPLATES ===
    elif emotion_label == "overwhelmed":
        templates = [
            # 0: paralysis from choice
            f'"{echo}" is decision paralysis. You\'re frozen because choosing one task feels like abandoning the others. But the paralysis itself is what\'s failing them all. There is no optimal sequence; there\'s only the cost of waiting for one to appear. Choose by deadline, by ease, or at random. Forward motion makes the next choice clearer. Pick one now.',

            # 1: volume overload
            f'"{echo}" is creating the illusion that you need to hold it all in your mind at once. You don\'t. Overwhelm is a capacity error: trying to process everything simultaneously instead of sequentially. Release everything but one. Do that one thing for 10 minutes as if nothing else exists. Let the rest wait in the dark.',

            # 2: holding all at once
            f'You\'re experiencing "{echo}" because you\'ve collapsed separate tasks into one crushing mass. Overwhelm happens when the boundaries between things dissolve. The antidote is artificial narrowing: write down three tasks, close the list, and do only the first one for 15 minutes. Shrink your world until it\'s survivable, then act inside it.',

            # 3: general overwhelmed
            f'Everything feels urgent because urgency is the emotion of overload, not a fact about your tasks. When everything screams for attention, none of them are actually louder, you just can\'t tell the difference. The way forward is arbitrary choice. Pick the task that would quiet the loudest inner voice, or the one you could finish fastest. Do that one badly. Let completion replace perfection.',
        ]
        return templates[template_idx]

    # === STRESSED TEMPLATES ===
    elif emotion_label == "stressed":
        templates = [
            # 0: time urgency
            f'"{echo}" is your body interpreting time as a closing fist. The scarcity creates tunnel vision that makes solutions invisible. But deadline pressure doesn\'t require perfection; it requires output. You don\'t need to feel capable. Just move. Set a 25-minute boundary and produce something rough. Speed, not quality. You can refine it later when the fist opens.',

            # 1: pressure reducing thinking
            f'"{echo}" is narrowing your thinking into a defensive crouch, which makes everything harder to see. Pressure doesn\'t sharpen focus; it collapses it. But you don\'t need expanded thinking right now. Try reducing scope. Choose the most obvious next action and do only that for 20 minutes. Let momentum replace clarity as the engine.',

            # 2: stress blocking cognition
            f'Stress is convincing you that confusion means incapacity, as if because you can\'t think clearly, you can\'t act at all. That\'s backward. Clarity is the result of action, not the prerequisite. Your hands don\'t need your mind\'s permission. Open the file and type for 10 minutes with zero expectation of coherence. Let motion generate the clarity stress is withholding.',

            # 3: general stressed
            f'Urgency is demanding immediate perfection, which is impossible and paralyzing. But stress distorts time. What feels like "everything now" is actually "something next, then something after." You don\'t need the whole task. Just take the next 15 minutes. Do the most obvious fragment badly. Survival output is still output. Polish is a luxury for after the deadline passes.',
        ]
        return templates[template_idx]

    # === AVOIDANT TEMPLATES ===
    elif emotion_label == "avoidant":
        templates = [
            # 0: start-stop cycle
            f'The pattern "{echo}" is a conditioning loop. Every retreat teaches the threat response to fire faster next time. Avoidance feels like self-protection but functions as self-training: the task becomes more dangerous with each escape. The loop breaks through exposure without completion. Open the file and sit with it for 5 minutes. Don\'t work. Just stop running. Teach yourself that proximity isn\'t danger.',

            # 1: explicit avoidance
            f'"{echo}" is the ache of knowing and not doing, and that gap is filled with shame disguised as relief. But avoidance doesn\'t reduce the discomfort; it compounds it with each delay. The task isn\'t getting easier by waiting; it\'s accumulating dread. Do the smallest possible fragment now: one sentence, one line, one click, without the burden of finishing. Just interrupt the pattern of retreat.',

            # 2: postponing
            f'The bargain "{echo}" is your mind trading present discomfort for the fantasy of future readiness. But later you will feel exactly what you feel now, plus the weight of continued delay. Motivation is the reward for starting, not the prerequisite. Set a 2-minute timer and do a miniature version of the task. Not well. Not completely. Just immediately. Two minutes to prove later is a lie.',

            # 3: distraction
            f'"{echo}" is the relief mechanism in real time. The discomfort of starting sends you toward anything softer, and the pattern strengthens each time you obey it. But distraction is debt: the task waits, and the next approach will be harder. Reverse the incentive. Do one small action on the task first: open one file, write one phrase, then earn the distraction after. Make avoidance cost something.',

            # 4: general avoidant
            f'You know what to do, but knowing isn\'t the obstacle. It\'s the absence of a feeling you\'re waiting for that will never arrive. Readiness, motivation, clarity: none of them precede action. They follow it. You don\'t need permission from your emotions to begin. Start in whatever state you\'re in, uncertain, resistant, afraid. Write one bad sentence. Let action create the feeling, not the reverse.',
        ]
        return templates[template_idx]

    # === DISCOURAGED TEMPLATES ===
    elif emotion_label == "discouraged":
        templates = [
            # 0: hopelessness
            f'When you think "{echo}", you\'re collapsing effort and outcome into the same thing, but they\'re not. Effort is yours; outcome is circumstance. Hopelessness mistakes uncertainty for inevitability. You can\'t see the result yet, but that doesn\'t mean trying is empty. It means trying is the only honest response to not knowing. Do 5 minutes of work as an experiment, not a solution. Test whether effort still holds meaning when success isn\'t promised.',

            # 1: absolutist thinking
            f'"{echo}" is the language of all-or-nothing, and absolutes erase the middle where most of living happens. You\'re frozen in a binary that doesn\'t exist: always failing or never capable. But patterns aren\'t permanent, and one moment doesn\'t predict all future ones. The story you\'re telling collapses time. Do one thing differently today, even trivially small, to prove the absolute is breakable. Let the exception shatter the rule.',

            # 2: low self-belief
            f'The belief "{echo}" is treating a current limitation as a fixed identity. But what you can\'t do yet isn\'t the same as what you are. The gap between where you are and where you want to be isn\'t evidence of inadequacy. It\'s the space where capability grows. You only see the gap because you can imagine better. Do one small imperfect thing today. Not to prove you\'re capable. To prove you can still try when you don\'t believe you are.',

            # 3: general discouraged
            f'You\'re reading the difficulty as a verdict. If it\'s hard, that doesn\'t mean you are not built for it. Struggle is information about the learning curve, not your ceiling. The absence of ease doesn\'t mean the presence of impossibility. Do the smallest version of the task available: one sentence, one step, one attempt, not to succeed, but to stay in contact with forward motion. Let trying be the point when winning feels out of reach.',
        ]
        return templates[template_idx]

    # === CALM TEMPLATES ===
    elif emotion_label == "calm":
        templates = [
            # 0: clarity
            f'"{echo}" is rare and worth protecting, but not by freezing or forcing output. Calm isn\'t a resource to extract; it\'s a state to inhabit. Take one gentle action from this place, not to justify the feeling, but because ease is what makes sustainable work possible. Let the calm inform the pace, not the urgency.',

            # 1: general calm
            f'This steadiness is valuable precisely because it asks nothing of you. You don\'t need to perform with it or push it into overdrive. Just notice what it feels like to not be fighting yourself. If something wants to be done, do it lightly. If nothing does, rest here. Let calm be its own destination, not a launching pad.',
        ]
        return templates[template_idx]

    # === TIRED TEMPLATES ===
    elif emotion_label == "tired":
        templates = [
            # 0: sleep deprivation
            f'"{echo}" is your body telling you a truth your expectations are ignoring. Sleep deprivation degrades cognition the same way alcohol does, but nobody would expect you to write an essay drunk. The honest move is to rest, or to do the absolute minimum: open the file for 2 minutes, then close it. Let rest be a strategy, not a failure.',

            # 1: burnout
            f'This exhaustion is not weakness. It is the cost of running at a pace your body could not sustain. You do not recover from burnout by pushing harder. You recover by doing less, deliberately and without guilt. If you can manage 5 minutes of the simplest possible task, do that. If not, rest. Tomorrow you will have more to work with than today.',

            # 2: general tired
            f'You are trying to think clearly with depleted resources, and your brain is rationing energy by making everything feel harder than it is. Fatigue distorts difficulty. The task is not as impossible as it feels right now. Do the smallest fragment you can: one sentence, one bullet point. Then stop. Tiny output from a tired mind still counts as forward motion.',
        ]
        return templates[template_idx]

    # === FRUSTRATED TEMPLATES ===
    elif emotion_label == "frustrated":
        templates = [
            # 0: self-directed anger
            f'The anger behind "{echo}" is pointed inward, and that means you still care. Frustration with yourself is not evidence of inadequacy; it is evidence that your standards exceed your current output. But punishing yourself for the gap does not close it. Channel the anger: pick one concrete action and do it roughly, imperfectly, with all that energy behind it. Let frustration become fuel instead of fire.',

            # 1: system frustration
            f'"{echo}" is the friction of things not cooperating. When systems break or tools fail, the rational response is irritation. But frustration narrows attention to the obstacle and hides the workaround. Step back for 60 seconds. Ask: what is the simplest path around this? Not the ideal fix, just the next move that bypasses what is broken. Redirect the energy.',

            # 2: repeated failure
            f'"{echo}" is the sting of a pattern repeating. Each time it happens, the frustration compounds because it feels like proof that nothing changes. But repeating the attempt is not the same as repeating the failure. Something is different each time, even if you cannot see it yet. Try one small variation: a different time, a different starting point, a different first step. Break the loop by changing one input.',

            # 3: general frustrated
            f'Frustration is impatience wearing a disguise. You want to be further ahead than you are, and the gap between expectation and reality feels intolerable. But the gap is not evidence of failure. It is the space where learning happens. Do one imperfect thing in the next 5 minutes. Not to satisfy the frustration, but to move through it. Action metabolises anger faster than thinking does.',
        ]
        return templates[template_idx]

    # === GUILTY TEMPLATES ===
    elif emotion_label == "guilty":
        templates = [
            # 0: letting others down
            f'The weight of "{echo}" is real, but guilt about letting someone down does not make you a bad person. It makes you someone who cares. The difference between guilt and action is movement: guilt keeps you frozen in the past, action moves you toward repair. You cannot undo yesterday. You can do one thing today that your future self will thank you for. Start there.',

            # 1: wasting time
            f'"{echo}" is guilt performing its favourite trick: making the lost time feel worse than it was by adding shame on top. But time you spent surviving, resting, or coping was not wasted. It was spent. The question now is not where the time went. It is what you can do with the next 10 minutes. One small action turns the narrative from loss to recovery.',

            # 2: should-have
            f'The thought "{echo}" is hindsight pretending to be wisdom. Of course you should have started earlier. You also should have been born knowing everything. Neither is possible. The version of you who delayed was doing their best with the resources they had. Guilt about the past does not earn you future productivity. Action does. Pick one thing, do it now, and let that be the turning point.',

            # 3: general guilty
            f'Guilt is a signal that your values and your actions are out of alignment, and that signal has value. But sitting in the guilt does not restore the alignment. Only action does. You do not need to fix everything at once. You need one step that moves you toward the person you want to be. Take that step now, however small, and let it quiet the guilt by one degree.',
        ]
        return templates[template_idx]

    # === HOPEFUL TEMPLATES ===
    elif emotion_label == "hopeful":
        templates = [
            # 0: evidence-based hope
            f'"{echo}" is your brain recognising real evidence that things are shifting. This is not blind optimism. It is pattern recognition: you did something, and something changed. The risk now is expecting too much too fast. Protect this feeling by keeping your next action small. One gentle step from this place. Let the hope carry you forward without overloading the moment.',

            # 1: general hopeful
            f'This feeling is fragile and worth protecting. Hope does not need to justify itself with massive action. It just needs one small thing to land on. Do something gentle from this place: review your notes, write one sentence, plan one step. Let the hope build naturally, without pressure to prove it was right. Momentum from ease lasts longer than momentum from urgency.',
        ]
        return templates[template_idx]

    # === PROUD TEMPLATES ===
    elif emotion_label == "proud":
        templates = [
            # 0: completion pride
            f'"{echo}" is a moment worth holding on to. You finished something, and that matters more than you think. The brain remembers completions as evidence of capability, and every completion makes the next start slightly easier. Take a moment to notice how this feels. Then, only if you want to, choose one small next thing. Let the pride carry you gently, not push you into overdrive.',

            # 1: general proud
            f'This pride is earned. You showed up, you did the thing, and that is evidence of who you are becoming. Do not rush past this feeling to start the next task. Sit with it. Let your brain register: I did this. I can do hard things. When you are ready, choose one small action from this place of strength. Not because you have to, but because you want to.',
        ]
        return templates[template_idx]

    # === NEUTRAL / OTHER ===
    else:
        return f'You wrote this down, which means part of you is still reaching toward change even when the path isn\'t clear. Awareness is the beginning of movement. You don\'t need a map or a plan. Just the next smallest step. Pick something near, something easy, or something unfinished. Do it for 10 minutes. Let action clarify what reflection can\'t.'


def get_belief_for_reframe(user_id):
    """Fetch a random identity belief from the user's stored beliefs."""
    if not user_id:
        return None
    try:
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT belief_text FROM identity_beliefs "
                "WHERE user_id = ? ORDER BY RANDOM() LIMIT 1",
                (user_id,),
            )
            row = cur.fetchone()
            if row and row["belief_text"] and row["belief_text"].strip():
                return row["belief_text"].strip()
            return None
        finally:
            conn.close()
    except Exception:
        # Edge case: DB error fetching belief
        return None


def personalise_reframe(reframe, emotion, behaviour, user_id):
    """
    Weave the user's identity belief into the reframe text (Layer 3).
    Takes a base reframe (already emotion + behaviour matched) and appends
    a belief-bridge sentence that connects the user's own words to the
    suggested action.
    Returns the enhanced reframe string, or the original if no belief found.
    """
    if not reframe:
        return reframe

    belief = get_belief_for_reframe(user_id)
    if not belief:
        # No belief available - return Layer 1 + Layer 2 only
        return reframe

    # Edge case: sanitise belief text to prevent format string issues
    safe_belief = belief.replace("{", "").replace("}", "")

    # Edge case: empty belief after sanitisation
    if not safe_belief.strip():
        return reframe

    # Ensure belief ends with a period for proper sentence flow
    if not safe_belief.endswith(('.', '!', '?')):
        safe_belief = safe_belief + '.'

    # Select the appropriate bridge based on behaviour (preferred) or emotion
    bridge = None
    if behaviour and behaviour in BELIEF_BRIDGES_BY_BEHAVIOUR:
        bridge = BELIEF_BRIDGES_BY_BEHAVIOUR[behaviour]
    elif emotion and emotion in BELIEF_BRIDGES_BY_EMOTION:
        bridge = BELIEF_BRIDGES_BY_EMOTION[emotion]
    else:
        # Edge case: unknown emotion and behaviour - generic bridge
        bridge = "Showing up like this is part of that."

    return f"{reframe} You said {safe_belief} {bridge}"


def extract_work_object(text: str) -> str:
    """
    Detect the specific work object (essay, report, dissertation, etc.)
    the user is writing about. Priority-ordered keyword matching.
    Returns a short label like 'essay' or 'task' (fallback).
    """
    import re
    text_lower = text.lower()

    work_objects = [
        (r'\b(dissertation|thesis)\b', 'dissertation'),
        (r'\b(report)\b', 'report'),
        (r'\b(essay)\b', 'essay'),
        (r'\b(assignment|homework|coursework)\b', 'assignment'),
        (r'\b(slides?|presentation|powerpoint)\b', 'slides'),
        (r'\b(lab|experiment)\b', 'lab'),
        (r'\b(reading|article|paper|textbook)\b', 'reading'),
        (r'\b(cod(?:e|ing)|program|project|app)\b', 'project'),
        (r'\b(notes?|study(?:ing)?|revision)\b', 'notes'),
        (r'\b(exam|test|quiz)\b', 'exam'),
        (r'\b(emails?|messages?)\b', 'email'),
        (r'\b(work|job)\b', 'work'),
    ]

    for pattern, label in work_objects:
        if re.search(pattern, text_lower):
            return label
    return 'task'


def generate_micro_task(journal_text: str, emotion_label: str) -> dict:
    """
    Generate a ≤2-minute micro-task that references the actual work object
    the user mentioned and is matched to their emotional state.

    Returns: { 'task_text': str, 'estimated_minutes': int, 'why_this': str }
    """
    work_obj = extract_work_object(journal_text)

    # map internal labels for micro-task compatibility
    if emotion_label == "stuck":
        emotion_label = "avoidant"
    elif emotion_label == "unmotivated":
        emotion_label = "discouraged"

    # Work-object-specific action fragments
    obj_actions = {
        'dissertation': {
            'open': 'Open your dissertation document',
            'write': 'Write one rough sentence in your dissertation',
            'small': 'Type one ugly heading in your dissertation',
            'review': 'Re-read the last paragraph of your dissertation',
        },
        'report': {
            'open': 'Open your report file',
            'write': 'Write one messy sentence in your report',
            'small': 'Add one bullet point to your report outline',
            'review': 'Re-read your report brief for 1 minute',
        },
        'essay': {
            'open': 'Open your essay document',
            'write': 'Write one bad sentence for your essay',
            'small': 'Type one rough idea for your essay argument',
            'review': 'Re-read the essay prompt for 1 minute',
        },
        'assignment': {
            'open': 'Open your assignment',
            'write': 'Write the first line of your assignment',
            'small': 'List 3 bullet points of what you need to cover',
            'review': 'Read the assignment brief for 1 minute',
        },
        'slides': {
            'open': 'Open your slide deck',
            'write': 'Write one slide title',
            'small': 'List 3 bullet points for your next slide',
            'review': 'Look through your slides for 1 minute',
        },
        'lab': {
            'open': 'Open your lab document',
            'write': 'Write one sentence in your lab write-up',
            'small': 'Jot down one observation from your experiment',
            'review': 'Read the lab instructions for 1 minute',
        },
        'reading': {
            'open': 'Open the article or paper',
            'write': 'Copy one quote from your reading into notes',
            'small': 'Read one page and highlight one sentence',
            'review': 'Skim the abstract and introduction headings',
        },
        'project': {
            'open': 'Open your project folder',
            'write': 'Write one line of code or one comment',
            'small': 'Add one TODO comment where you left off',
            'review': 'Read through the last 10 lines you wrote',
        },
        'notes': {
            'open': 'Open your notes',
            'write': 'Write one bullet point in your notes',
            'small': 'Write down one thing you remember without checking',
            'review': 'Re-read one page of your notes',
        },
        'exam': {
            'open': 'Open your study materials',
            'write': 'Write one flashcard from memory',
            'small': 'Write down 3 key terms you need to know',
            'review': 'Read one page of your notes aloud',
        },
        'email': {
            'open': 'Open your email draft',
            'write': 'Write the subject line and first sentence',
            'small': 'Draft 2 bullet points of what you want to say',
            'review': 'Re-read the email you need to reply to',
        },
        'work': {
            'open': 'Open the file you need to work on',
            'write': 'Write one sentence toward your task',
            'small': 'Jot down the single next step on paper',
            'review': 'Review your task list for 1 minute',
        },
        'task': {
            'open': 'Open the file you have been avoiding',
            'write': 'Write one rough sentence on whatever is due next',
            'small': 'Write down the single next step on paper',
            'review': 'Spend 1 minute reviewing what needs doing',
        },
    }

    actions = obj_actions.get(work_obj, obj_actions['task'])

    if emotion_label == "anxious":
        return {
            'task_text': f"{actions['open']} and leave it visible for 2 minutes. Don't type. Just sit with it open.",
            'estimated_minutes': 2,
            'why_this': f"Anxiety spikes at the threshold of starting. Having your {work_obj} open without pressure reduces that spike over time.",
        }
    elif emotion_label == "overwhelmed":
        return {
            'task_text': f"{actions['small']}. Just that one thing. Ignore everything else for now.",
            'estimated_minutes': 1,
            'why_this': f"Overwhelm dissolves when you shrink scope to a single visible action on your {work_obj}.",
        }
    elif emotion_label == "stressed":
        return {
            'task_text': f"{actions['write']}, messy and imperfect. Speed over quality for 2 minutes.",
            'estimated_minutes': 2,
            'why_this': f"Under time pressure, rough output on your {work_obj} beats no output.",
        }
    elif emotion_label == "avoidant":
        return {
            'task_text': f"{actions['open']}. Set a 2-minute timer. When it rings, you can close it.",
            'estimated_minutes': 2,
            'why_this': f"Avoidance strengthens with distance. Two minutes near your {work_obj} weakens the pattern.",
        }
    elif emotion_label == "discouraged":
        return {
            'task_text': f"{actions['write']}, not to finish, just to prove movement is still possible.",
            'estimated_minutes': 1,
            'why_this': f"Discouragement says effort is pointless. One sentence in your {work_obj} says otherwise.",
        }
    elif emotion_label == "calm":
        return {
            'task_text': f"{actions['review']} and note one thing you want to work on next.",
            'estimated_minutes': 2,
            'why_this': f"Calm is the best state to gently re-engage with your {work_obj} without pressure.",
        }
    elif emotion_label == "tired":
        return {
            'task_text': f"{actions['open']} and look at it for 1 minute. If energy allows, {actions['small'].lower()}. If not, close it and rest.",
            'estimated_minutes': 1,
            'why_this': f"When energy is low, even opening your {work_obj} counts. Pushing harder backfires when tired.",
        }
    elif emotion_label == "frustrated":
        return {
            'task_text': f"{actions['write']}, rough and imperfect. Channel the frustration into output for 2 minutes.",
            'estimated_minutes': 2,
            'why_this': f"Frustration carries energy. Two minutes of rough output on your {work_obj} converts anger into momentum.",
        }
    elif emotion_label == "guilty":
        return {
            'task_text': f"{actions['small']}. One small action now is worth more than guilt about yesterday.",
            'estimated_minutes': 1,
            'why_this': f"Guilt locks you in the past. One action on your {work_obj} moves you into the present.",
        }
    elif emotion_label == "hopeful":
        return {
            'task_text': f"{actions['review']} and write down one thing you want to do next. Keep it gentle.",
            'estimated_minutes': 2,
            'why_this': f"Hope is a good state to plan from. A gentle step on your {work_obj} builds on the feeling without forcing it.",
        }
    elif emotion_label == "proud":
        return {
            'task_text': f"Take a moment to notice what you accomplished. Then, if you want, {actions['review'].lower()} and pick one small next step.",
            'estimated_minutes': 2,
            'why_this': f"Pride is evidence your brain is rewiring. One gentle step on your {work_obj} from this place strengthens the pattern.",
        }
    else:
        return {
            'task_text': f"{actions['open']} and spend 2 minutes looking at where you left off.",
            'estimated_minutes': 2,
            'why_this': f"Re-engaging with your {work_obj} for 2 minutes builds momentum without pressure.",
        }


# Combined intervention matrix - Task 2.3
# Maps (emotion, behaviour) to a targeted reframe + micro-task
# selected from BOTH dimensions of the ML pipeline.
# Template placeholders:
#   {echo}     - phrase echoed from the user's journal text
#   {work_obj} - detected work object (essay, report, task, etc.)

INTERVENTION_MATRIX = {

    # -- AVOIDANCE: user is distancing from or putting off work --

    ("overwhelmed", "avoidance"): {
        "reframe": 'You wrote "{echo}" and the sheer volume pushed you into retreat. Your brain is creating distance from something that feels too big. You do not need to face all of it right now, just one piece.',
        "task_text": "Open your {work_obj} and look at just one section for 2 minutes. Ignore everything else.",
        "task_minutes": 2,
        "why_this": "Overwhelm-driven avoidance breaks when you narrow scope to one visible piece of your {work_obj}.",
    },
    ("anxious", "avoidance"): {
        "reframe": 'The avoidance around "{echo}" is shielding you from uncertainty, but each retreat teaches your brain the work is dangerous. It is not. The danger is a feeling, not a fact.',
        "task_text": "Open your {work_obj} and leave it visible for 2 minutes. Do not type, just let it be open.",
        "task_minutes": 2,
        "why_this": "Anxiety-driven avoidance weakens through safe, pressure-free proximity to your {work_obj}.",
    },
    ("stuck", "avoidance"): {
        "reframe": 'You feel stuck around "{echo}" and the not-knowing has turned into not-starting. Being stuck is not the same as being unable. The path forward appears when you get close enough to see it.',
        "task_text": "Open your {work_obj} and set a 2-minute timer. Sit with it open. When the timer ends, you can close it.",
        "task_minutes": 2,
        "why_this": "When stuckness becomes avoidance, proximity to your {work_obj} is the first step back.",
    },
    ("stressed", "avoidance"): {
        "reframe": 'Time pressure around "{echo}" makes avoidance feel rational, as if waiting might somehow create more time. It will not. The task is not getting easier and the clock does not stop.',
        "task_text": "Open your {work_obj} and write one rough sentence. Speed over quality for 2 minutes.",
        "task_minutes": 2,
        "why_this": "Stress-driven avoidance burns the time that pressure claims to protect. One rough action on your {work_obj} is enough.",
    },
    ("tired", "avoidance"): {
        "reframe": 'Your energy is low and avoiding "{echo}" feels like self-care. Sometimes rest is the right call. But if this is weighing on you, even 1 minute of contact is better than carrying the weight of avoidance.',
        "task_text": "Open your {work_obj} for 1 minute. If nothing comes, close it and rest without guilt.",
        "task_minutes": 1,
        "why_this": "When tired, brief contact with your {work_obj} breaks avoidance without draining energy.",
    },
    ("calm", "avoidance"): {
        "reframe": 'You feel steady, but something about "{echo}" is still being avoided. This calm is the best possible state to approach what you have been putting off. The work will feel less threatening from here.',
        "task_text": "Open your {work_obj} gently and spend 2 minutes with it. Let the calm carry you in.",
        "task_minutes": 2,
        "why_this": "Calm is the safest state to re-engage with your {work_obj} and break the avoidance pattern.",
    },
    ("frustrated", "avoidance"): {
        "reframe": 'The frustration behind "{echo}" is pushing you away from the work, but that anger carries energy. Right now it is fuelling retreat. It could fuel movement instead.',
        "task_text": "Open your {work_obj} and channel 2 minutes of that energy into rough, imperfect output.",
        "task_minutes": 2,
        "why_this": "Frustration contains energy that, redirected toward your {work_obj}, converts anger into momentum.",
    },
    ("guilty", "avoidance"): {
        "reframe": 'Guilt about "{echo}" is adding weight to the avoidance. Each hour of delay layers more shame on top, making the next start harder. One small action now breaks the loop that guilt and avoidance feed together.',
        "task_text": "Open your {work_obj} and do one small thing. One sentence, one bullet point, anything.",
        "task_minutes": 1,
        "why_this": "Guilt and avoidance feed each other. One action on your {work_obj} interrupts both.",
    },
    ("unmotivated", "avoidance"): {
        "reframe": 'Nothing about "{echo}" feels worth the effort, and the avoidance feels like honesty. But motivation is not a prerequisite for action. It is a byproduct of it. Contact with the work creates the drive that waiting never will.',
        "task_text": "Open your {work_obj} and spend 2 minutes looking at where you left off. Let contact replace enthusiasm.",
        "task_minutes": 2,
        "why_this": "When unmotivated and avoiding, proximity to your {work_obj} can generate the motivation waiting cannot.",
    },
    ("hopeful", "avoidance"): {
        "reframe": 'The hope you feel is real, but "{echo}" suggests something is still being held at a distance. This is a good moment to close the gap between feeling ready and actually starting.',
        "task_text": "Open your {work_obj} and take one gentle step from this place of hope. Let it carry you in.",
        "task_minutes": 2,
        "why_this": "Hope paired with avoidance means the readiness is there but the first step is missing on your {work_obj}.",
    },
    ("proud", "avoidance"): {
        "reframe": 'You have something to be proud of, but "{echo}" tells me something else is being avoided. Use the confidence from what you accomplished. You already know you can do hard things.',
        "task_text": "Open your {work_obj} and carry this momentum into one small action. 2 minutes.",
        "task_minutes": 2,
        "why_this": "Pride builds confidence that can overcome avoidance of your {work_obj}.",
    },

    # -- OVERWHELM: user is drowning in volume or scope --

    ("overwhelmed", "overwhelm"): {
        "reframe": 'Both your feeling and your behaviour around "{echo}" are telling the same story: there is too much, and you are trying to hold it all at once. You cannot process everything simultaneously. Release everything but one task.',
        "task_text": "Write down three things related to your {work_obj}. Pick the smallest. Do just that for 2 minutes.",
        "task_minutes": 2,
        "why_this": "When overwhelm is both feeling and behaviour, narrowing to one piece of your {work_obj} is the only way through.",
    },
    ("anxious", "overwhelm"): {
        "reframe": 'Anxiety is making every part of "{echo}" feel equally urgent, and the volume is genuine. But urgency is a feeling, not a fact about your tasks. Not everything needs to happen now.',
        "task_text": "Pick the one part of your {work_obj} that would quiet the loudest worry. Give it 2 minutes.",
        "task_minutes": 2,
        "why_this": "Anxiety amplifies overwhelm. Picking one worry-reducing action on your {work_obj} calms both.",
    },
    ("stuck", "overwhelm"): {
        "reframe": 'You cannot find a way through "{echo}" because you are looking at everything at once. The path is hidden behind the volume. Narrow your view to one single task and the rest will wait.',
        "task_text": "Pick one section of your {work_obj} and spend 2 minutes on just that. Ignore the rest.",
        "task_minutes": 2,
        "why_this": "Stuckness from overwhelm resolves when you limit your view to one piece of your {work_obj}.",
    },
    ("stressed", "overwhelm"): {
        "reframe": 'Deadline pressure around "{echo}" is making everything feel simultaneous. Time scarcity and volume are a difficult combination, but they both respond to the same fix: pick one thing.',
        "task_text": "Pick the most urgent part of your {work_obj} and give it 2 minutes of rough, imperfect attention.",
        "task_minutes": 2,
        "why_this": "Stress plus overwhelm requires forced prioritisation of one part of your {work_obj}.",
    },
    ("tired", "overwhelm"): {
        "reframe": 'Your energy is too low to process this volume around "{echo}". Fatigue makes everything look bigger than it is. You do not have the capacity for all of it, and that is fine. You have the capacity for one thing.',
        "task_text": "Pick the easiest part of your {work_obj} and spend 1 minute on it. Protect your energy.",
        "task_minutes": 1,
        "why_this": "Tired and overwhelmed means your {work_obj} needs the easiest possible entry point.",
    },
    ("calm", "overwhelm"): {
        "reframe": 'You feel calm but the volume around "{echo}" is real. This steadiness is your advantage right now. Most people face overwhelm from anxiety or stress. You get to face it from stability.',
        "task_text": "Use this calm to pick one part of your {work_obj} and give it 2 minutes of gentle attention.",
        "task_minutes": 2,
        "why_this": "Calm during overwhelm is rare. Use it to make one clear choice about your {work_obj}.",
    },
    ("frustrated", "overwhelm"): {
        "reframe": 'The volume around "{echo}" is real and the frustration makes it feel worse. But that energy can be useful. Instead of fighting the volume, pick one task that would feel satisfying to finish.',
        "task_text": "Pick one part of your {work_obj} that would feel good to finish and give it 2 rough minutes.",
        "task_minutes": 2,
        "why_this": "Frustration contains energy that, aimed at one piece of your {work_obj}, cuts through overwhelm.",
    },
    ("guilty", "overwhelm"): {
        "reframe": 'Guilt about "{echo}" is adding emotional weight on top of the practical volume. That combination is crushing. Release the guilt for now. It does not help you process the work.',
        "task_text": "Write down one thing you can do on your {work_obj} in the next 2 minutes and do only that.",
        "task_minutes": 2,
        "why_this": "Guilt amplifies overwhelm. One concrete action on your {work_obj} quiets both.",
    },
    ("unmotivated", "overwhelm"): {
        "reframe": 'Nothing feels worth starting when there is this much going on with "{echo}". The volume drains motivation before you even begin. Pick the task that requires the least energy.',
        "task_text": "Pick the easiest part of your {work_obj} and spend 2 minutes on it. Start with what is simple.",
        "task_minutes": 2,
        "why_this": "Unmotivated plus overwhelmed means your {work_obj} needs the lowest-effort entry point.",
    },
    ("hopeful", "overwhelm"): {
        "reframe": 'The hope you feel is a good sign even with all this volume around "{echo}". Let the hope guide your choice. Pick the one task that feels most possible right now.',
        "task_text": "Pick the part of your {work_obj} that excites you most and give it 2 gentle minutes.",
        "task_minutes": 2,
        "why_this": "Hope during overwhelm is a compass. Use it to pick the right piece of your {work_obj}.",
    },
    ("proud", "overwhelm"): {
        "reframe": 'You accomplished something worth being proud of, and the remaining volume of "{echo}" is still there. Let the pride sit. You have already proven you can do hard things. One more small thing.',
        "task_text": "Pick one remaining part of your {work_obj} and give it 2 minutes. You already proved you can.",
        "task_minutes": 2,
        "why_this": "Pride after accomplishment provides evidence you can handle one more piece of your {work_obj}.",
    },

    # -- RUMINATION: user is stuck in thought loops, overthinking --

    ("overwhelmed", "rumination"): {
        "reframe": 'Your mind is spinning about "{echo}" without finding a way forward. Thinking harder about too many things will not produce clarity. The loop needs to be broken externally.',
        "task_text": "Write down what is on your mind about your {work_obj} on paper. Then set a 2-minute timer and do one physical action.",
        "task_minutes": 2,
        "why_this": "Overwhelm fuels rumination. Externalising thoughts about your {work_obj} breaks the loop.",
    },
    ("anxious", "rumination"): {
        "reframe": 'The worry loop around "{echo}" is your brain rehearsing danger that may not arrive. This loop consumes energy without producing safety. The way out is not more thinking, it is one action.',
        "task_text": "Write down the worst case about your {work_obj}, then the most likely case. Then do one small action.",
        "task_minutes": 2,
        "why_this": "Anxious rumination responds to reality-testing followed by one concrete step on your {work_obj}.",
    },
    ("stuck", "rumination"): {
        "reframe": 'You are thinking in circles about "{echo}" and waiting for clarity to appear. It will not come from more thinking. Clarity is a byproduct of action, not a prerequisite for it.',
        "task_text": "Set a 2-minute timer and write one sentence about your {work_obj}. Any sentence. Break the loop.",
        "task_minutes": 2,
        "why_this": "Stuck rumination breaks when you force one output on your {work_obj}, however imperfect.",
    },
    ("stressed", "rumination"): {
        "reframe": 'Time pressure has your mind racing about "{echo}" without landing anywhere. The racing is burning time, not saving it. Thinking faster does not equal thinking better.',
        "task_text": "Set a 2-minute timer. Do the most obvious thing on your {work_obj}. Rough output is still output.",
        "task_minutes": 2,
        "why_this": "Stress-driven rumination wastes the time it claims to protect. One rough action on your {work_obj} is better.",
    },
    ("tired", "rumination"): {
        "reframe": 'Your mind is circling "{echo}" but you are too tired to think productively. The loop is consuming energy you do not have. Let it go for now.',
        "task_text": "Write down the one thought about your {work_obj} that keeps repeating. Then close everything and rest.",
        "task_minutes": 1,
        "why_this": "Tired rumination drains the little energy left. Capture the thought about your {work_obj} and rest.",
    },
    ("calm", "rumination"): {
        "reframe": 'You feel calm but your mind is still circling "{echo}". Use this clarity to put the thoughts on paper. Once externalised, the loop loses its grip on your attention.',
        "task_text": "Write down what you are thinking about your {work_obj}. Then take one small action on it.",
        "task_minutes": 2,
        "why_this": "Calm rumination is the easiest loop to break. Externalise thoughts about your {work_obj} and act.",
    },
    ("frustrated", "rumination"): {
        "reframe": 'Frustration about "{echo}" is feeding the thought loop, and the loop is feeding the frustration. They are amplifying each other. The cycle breaks when you do something different.',
        "task_text": "Write down what is frustrating about your {work_obj}. Then try one thing differently from last time.",
        "task_minutes": 2,
        "why_this": "Frustrated rumination feeds itself. One variation in approach to your {work_obj} interrupts the loop.",
    },
    ("guilty", "rumination"): {
        "reframe": 'The guilt about "{echo}" is replaying on a loop, and each replay makes it heavier. Guilt in your head compounds. Guilt addressed with action dissolves.',
        "task_text": "Write down what you feel guilty about regarding your {work_obj}. Then do one thing that moves forward.",
        "task_minutes": 2,
        "why_this": "Guilty rumination dissolves with one forward action on your {work_obj}.",
    },
    ("unmotivated", "rumination"): {
        "reframe": 'You are stuck thinking about "{echo}" without any drive to act on it. The loop will not generate motivation. Motion generates motivation, not the reverse.',
        "task_text": "Write down one thing you could do on your {work_obj} in 2 minutes. Then do it. Motion first.",
        "task_minutes": 2,
        "why_this": "Unmotivated rumination waits for motivation that only comes from acting on your {work_obj}.",
    },
    ("hopeful", "rumination"): {
        "reframe": 'Hope is present but your mind is still processing "{echo}". The thinking is circling around possibility without landing. Let the hope move from thought into one real action.',
        "task_text": "Write down what you are hoping for with your {work_obj}. Then take one small step toward it.",
        "task_minutes": 2,
        "why_this": "Hopeful rumination resolves when the hope becomes one concrete action on your {work_obj}.",
    },
    ("proud", "rumination"): {
        "reframe": 'You have something to be proud of, but your mind is still circling "{echo}". Write down what you accomplished. Let yourself see it on paper instead of spinning it in your head.',
        "task_text": "Write down what you accomplished with your {work_obj}. Then, if you want, plan one next step.",
        "task_minutes": 2,
        "why_this": "Proud rumination resolves when the accomplishment is externalised and your {work_obj} progress is visible.",
    },

    # -- ACTION: user is actively working (support continuation) --

    ("overwhelmed", "action"): {
        "reframe": 'You are moving forward on "{echo}" despite feeling overwhelmed. That takes real strength. The overwhelm has not stopped you. Keep your focus narrow and your pace steady.',
        "task_text": "Keep going with your {work_obj} for 2 more minutes. If the overwhelm rises, shrink your focus smaller.",
        "task_minutes": 2,
        "why_this": "Action during overwhelm is rare and valuable. Protecting momentum on your {work_obj} matters more than speed.",
    },
    ("anxious", "action"): {
        "reframe": 'You are working through the anxiety around "{echo}". That is harder than most people understand. The anxiety is loud, but you are moving anyway. That matters.',
        "task_text": "Keep your current pace on your {work_obj}. If anxiety spikes, pause for 30 seconds, breathe, then continue.",
        "task_minutes": 2,
        "why_this": "Anxious action is courageous. Steady pace on your {work_obj} teaches your brain the work is safe.",
    },
    ("stuck", "action"): {
        "reframe": 'You were stuck on "{echo}" but you found a way through. Do not question why it is working. Just stay with this pace. If stuckness returns, make the task even smaller.',
        "task_text": "Keep going with your {work_obj} at this pace for 2 more minutes. You are already moving.",
        "task_minutes": 2,
        "why_this": "Movement after stuckness is fragile. Protecting momentum on your {work_obj} is the priority.",
    },
    ("stressed", "action"): {
        "reframe": 'You are producing output under pressure around "{echo}". Rough and fast is fine right now. Refinement comes later. What matters is that something is being created.',
        "task_text": "Keep working on your {work_obj} for 2 more minutes. Speed over quality. You can refine later.",
        "task_minutes": 2,
        "why_this": "Stressed action produces output. Rough progress on your {work_obj} beats no progress.",
    },
    ("tired", "action"): {
        "reframe": 'You are working on "{echo}" despite low energy. That is admirable, but be careful not to push past your limit. Sustainable effort lasts longer than desperate sprints.',
        "task_text": "Keep going with your {work_obj} for 2 more minutes, then take a real break. Protect your energy.",
        "task_minutes": 2,
        "why_this": "Tired action needs a boundary. Two more minutes on your {work_obj}, then genuine rest.",
    },
    ("calm", "action"): {
        "reframe": 'You are in a good space with "{echo}". This is what sustainable productivity feels like. Protect this state by keeping your pace gentle and your expectations reasonable.',
        "task_text": "Keep working on your {work_obj} at this easy pace. No need to push harder. Steady wins.",
        "task_minutes": 2,
        "why_this": "Calm action is the most sustainable state for your {work_obj}. Protect it by not overcommitting.",
    },
    ("frustrated", "action"): {
        "reframe": 'The frustration around "{echo}" is fuelling your output. That energy is useful right now. Channel it, but watch for the moment when anger stops being productive and starts making mistakes.',
        "task_text": "Keep going on your {work_obj} for 2 more minutes. When the frustration shifts to exhaustion, pause.",
        "task_minutes": 2,
        "why_this": "Frustrated action has energy but limited runway. Use it wisely on your {work_obj}.",
    },
    ("guilty", "action"): {
        "reframe": 'You are working through the guilt of "{echo}". Every minute you put in now is closing the gap between where you are and where you want to be. Let the action quiet the guilt.',
        "task_text": "Keep going on your {work_obj} for 2 more minutes. Each minute of work dissolves a layer of guilt.",
        "task_minutes": 2,
        "why_this": "Guilty action is restorative. Each minute on your {work_obj} reduces the gap guilt points at.",
    },
    ("unmotivated", "action"): {
        "reframe": 'You are working on "{echo}" without motivation. That is the hardest kind of effort and it counts the most. Motivation follows action. You are building it right now.',
        "task_text": "Keep going on your {work_obj} for 2 more minutes. You are proving motivation is not required.",
        "task_minutes": 2,
        "why_this": "Unmotivated action is discipline. Each minute on your {work_obj} generates the drive waiting could not.",
    },
    ("hopeful", "action"): {
        "reframe": 'Hope and action together on "{echo}" are building real momentum. This is what change feels like from the inside. Let the hope sustain your pace without pushing into overdrive.',
        "task_text": "Keep working on your {work_obj} gently. Let the hope carry the pace. No need to rush.",
        "task_minutes": 2,
        "why_this": "Hopeful action builds on itself. Gentle momentum on your {work_obj} lasts longer than urgency.",
    },
    ("proud", "action"): {
        "reframe": 'You are working from a place of pride and momentum on "{echo}". This is the ideal state. Protect your energy. When pride shifts to pressure, take a break.',
        "task_text": "Keep working on your {work_obj} from this place of strength. When you feel the shift to pressure, stop and rest.",
        "task_minutes": 2,
        "why_this": "Proud action is powerful but can tip into overcommitment. Protect your energy on your {work_obj}.",
    },

    # -- COMPLETION: user just finished something (celebrate, rest, do not push) --

    ("overwhelmed", "completion"): {
        "reframe": 'You finished something related to "{echo}" even while feeling overwhelmed. That matters more than you realise. Rest here for a moment. Do not immediately look at the next thing on the pile.',
        "task_text": "Take a moment to notice what you finished on your {work_obj}. Close it. Rest before looking at what is next.",
        "task_minutes": 1,
        "why_this": "Completion during overwhelm needs acknowledgment. Rushing to the next part of your {work_obj} erases the win.",
    },
    ("anxious", "completion"): {
        "reframe": 'You completed something on "{echo}" despite the anxiety telling you it would go wrong. It did not. You did it. Sit with that for a moment before the anxiety tries to pull you into the next worry.',
        "task_text": "Notice what you finished on your {work_obj}. The next task can wait. Let your brain register this win.",
        "task_minutes": 1,
        "why_this": "Anxious completion needs space. Pausing after your {work_obj} win teaches your brain it was safe.",
    },
    ("stuck", "completion"): {
        "reframe": 'You were stuck on "{echo}" and you still managed to finish something. That is real evidence that being stuck is not permanent. Take a moment to notice how this feels.',
        "task_text": "Notice what you finished on your {work_obj}. Being stuck did not stop you. Remember that.",
        "task_minutes": 1,
        "why_this": "Completion after stuckness is evidence that stuckness on your {work_obj} is temporary.",
    },
    ("stressed", "completion"): {
        "reframe": 'You finished something on "{echo}" under pressure. The stress made it feel harder than it was. Take a breath. You do not need to carry the urgency into the next task.',
        "task_text": "Take a breath. You finished part of your {work_obj}. Let the urgency ease before starting the next part.",
        "task_minutes": 1,
        "why_this": "Stressed completion needs decompression. A pause after your {work_obj} prevents burnout.",
    },
    ("tired", "completion"): {
        "reframe": 'You finished something on "{echo}" despite low energy. Rest now. You have earned it. Do not start the next thing until your energy recovers.',
        "task_text": "You finished part of your {work_obj}. Close it. Rest. The next task will be easier when your energy returns.",
        "task_minutes": 1,
        "why_this": "Tired completion demands rest. Recovery now makes your next session on your {work_obj} more effective.",
    },
    ("calm", "completion"): {
        "reframe": 'You completed something on "{echo}" from a place of calm. This is what sustainable progress feels like. No urgency, no panic, just steady output. Notice how different this feels.',
        "task_text": "Notice what you finished on your {work_obj}. Rest or continue gently, whichever feels right. No pressure.",
        "task_minutes": 1,
        "why_this": "Calm completion is the ideal state. Savouring it trains your brain that your {work_obj} can feel good.",
    },
    ("frustrated", "completion"): {
        "reframe": 'You pushed through the frustration of "{echo}" and finished something. Let the frustration go now. It served its purpose. You did the thing you were angry about.',
        "task_text": "Notice what you finished on your {work_obj}. The frustration can release now. It did its job.",
        "task_minutes": 1,
        "why_this": "Frustrated completion deserves acknowledgment. Let the anger go now that your {work_obj} is done.",
    },
    ("guilty", "completion"): {
        "reframe": 'You finished something that guilt about "{echo}" was weighing on. Let the guilt release. One action was all it took. The gap between your values and your actions just got smaller.',
        "task_text": "Notice what you finished on your {work_obj}. The guilt can ease now. You took action.",
        "task_minutes": 1,
        "why_this": "Guilty completion restores alignment. Finishing part of your {work_obj} proves action is possible.",
    },
    ("unmotivated", "completion"): {
        "reframe": 'You finished something about "{echo}" without motivation driving you. That is discipline in its purest form. It is more reliable than motivation will ever be. Rest now.',
        "task_text": "You finished part of your {work_obj} without motivation. That is worth noticing. Rest.",
        "task_minutes": 1,
        "why_this": "Unmotivated completion is evidence of discipline. Your {work_obj} does not require motivation.",
    },
    ("hopeful", "completion"): {
        "reframe": 'You completed something on "{echo}" and the hope was justified. This is evidence that things are changing. Let your brain register this before rushing to the next thing.',
        "task_text": "Notice what you finished on your {work_obj}. The hope was right. Sit with this feeling.",
        "task_minutes": 1,
        "why_this": "Hopeful completion reinforces the hope. Let your brain register this {work_obj} win.",
    },
    ("proud", "completion"): {
        "reframe": 'You did something hard with "{echo}" and you are proud. That is evidence of who you are becoming. Rest. Recovery is part of the process. You do not need to earn this feeling by doing more.',
        "task_text": "You finished part of your {work_obj}. Take this pride with you. Rest. Do not immediately push for more.",
        "task_minutes": 1,
        "why_this": "Proud completion is the moment to rest. Your {work_obj} progress is real and does not need proving.",
    },

    # -- RECOVERY: user is resting or recovering from effort --

    ("overwhelmed", "recovery"): {
        "reframe": 'You are recovering from overwhelm around "{echo}". This is not avoidance, this is necessary. Your brain needs space to process the volume before it can act on it.',
        "task_text": "Rest. When you feel even slightly ready, open your {work_obj} and look at one small part for 1 minute.",
        "task_minutes": 1,
        "why_this": "Recovery after overwhelm is not weakness. Returning to your {work_obj} gently prevents relapse.",
    },
    ("anxious", "recovery"): {
        "reframe": 'The anxiety around "{echo}" is still present but you are giving yourself space. Rest is part of the process. When a small window of calm appears, you can use it for one tiny action.',
        "task_text": "Rest. When the anxiety eases slightly, open your {work_obj} for 1 minute. Until then, be here.",
        "task_minutes": 1,
        "why_this": "Anxious recovery needs patience. Gentle re-contact with your {work_obj} when calm appears.",
    },
    ("stuck", "recovery"): {
        "reframe": 'You are resting after a period of stuckness on "{echo}". Sometimes the best thing for a stuck mind is to stop trying. The subconscious often finds the path when the conscious mind stops forcing it.',
        "task_text": "Rest fully. When you return to your {work_obj}, try a completely different starting point.",
        "task_minutes": 1,
        "why_this": "Rest after stuckness lets the subconscious work on your {work_obj} in the background.",
    },
    ("stressed", "recovery"): {
        "reframe": 'You are recovering from stress around "{echo}". The urgency is lying to you right now. You need this break. When you return, start with the easiest task first.',
        "task_text": "Rest now. When you return to your {work_obj}, start with the easiest part. Do not re-enter through urgency.",
        "task_minutes": 1,
        "why_this": "Stressed recovery requires deliberate re-entry. Start with the easiest part of your {work_obj}.",
    },
    ("tired", "recovery"): {
        "reframe": 'Rest is exactly what you need right now around "{echo}". This is not failure. This is your body telling you a truth your expectations are ignoring. Close everything and recover.',
        "task_text": "Close your {work_obj}. Rest. The work will be easier when you return with energy.",
        "task_minutes": 1,
        "why_this": "Tired recovery is non-negotiable. Your {work_obj} benefits more from rest than from forced output.",
    },
    ("calm", "recovery"): {
        "reframe": 'You are in a calm, restful state around "{echo}". Protect this. Do not let urgency pull you out before you are ready. When you choose to return, start gently.',
        "task_text": "Stay in this calm. When you choose to return to your {work_obj}, take one gentle step. No pressure.",
        "task_minutes": 1,
        "why_this": "Calm recovery is ideal. Protect it and return to your {work_obj} when genuinely ready.",
    },
    ("frustrated", "recovery"): {
        "reframe": 'You stepped back from frustration around "{echo}". That was a good call. The emotion was starting to work against you. Let it settle before returning.',
        "task_text": "Rest until the frustration eases. When you return to your {work_obj}, try a different approach.",
        "task_minutes": 1,
        "why_this": "Recovery from frustrated work prevents your {work_obj} from becoming associated with anger.",
    },
    ("guilty", "recovery"): {
        "reframe": 'Guilt is telling you that resting from "{echo}" is wrong. It is not. Recovery is not laziness. It is how you become capable of the next effort. Rest now and return without the weight of shame.',
        "task_text": "Rest without guilt. When you return to your {work_obj}, one small action is enough to start.",
        "task_minutes": 1,
        "why_this": "Guilty recovery needs permission. Rest makes your next session on your {work_obj} better.",
    },
    ("unmotivated", "recovery"): {
        "reframe": 'You have no drive around "{echo}" and you are resting. That is honest. Forcing motivation when there is none creates resistance. If a small spark appears, follow it. If not, that is okay for now.',
        "task_text": "Rest. If a small spark of interest in your {work_obj} appears, follow it gently. If not, rest fully.",
        "task_minutes": 1,
        "why_this": "Unmotivated recovery should be free of pressure. Your {work_obj} will still be there.",
    },
    ("hopeful", "recovery"): {
        "reframe": 'The hope around "{echo}" is there and so is the need to rest. You do not need to act on the hope right now. Recovery will give you the energy to act on it when you are ready.',
        "task_text": "Rest. The hope about your {work_obj} will be here when you return. Recovery strengthens it.",
        "task_minutes": 1,
        "why_this": "Hopeful recovery lets the hope grow. Rest gives energy to act on your {work_obj} later.",
    },
    ("proud", "recovery"): {
        "reframe": 'You have earned this rest after "{echo}". Do not skip recovery to chase the next thing. Pride that leads to burnout is not sustainable. Rest fully. The next effort will be better for it.',
        "task_text": "Rest. You earned it with your {work_obj}. The next session will be stronger after real recovery.",
        "task_minutes": 1,
        "why_this": "Proud recovery prevents the pride-to-burnout cycle. Your {work_obj} benefits from genuine rest.",
    },
}


def get_combined_intervention(journal_text, emotion, behaviour, user_id):
    """Select intervention from emotion x behaviour matrix. Returns None if behaviour unavailable."""

    # Edge case 1: behaviour model not loaded or prediction failed
    if behaviour is None:
        return None

    # Edge case 2: empty or missing journal text
    if not journal_text or not journal_text.strip():
        return None

    # Edge case 3: unexpected emotion or behaviour label from model
    valid_emotions = [
        "overwhelmed", "anxious", "stuck", "stressed", "tired", "calm",
        "frustrated", "guilty", "unmotivated", "hopeful", "proud",
    ]
    valid_behaviours = [
        "avoidance", "overwhelm", "action", "completion", "recovery", "rumination",
    ]
    if emotion not in valid_emotions or behaviour not in valid_behaviours:
        return None

    # Extract echo phrase from journal text (with fallback)
    try:
        echo = extract_echo_phrase(journal_text)
    except Exception:
        # Edge case 4: echo extraction fails on unusual input
        first_sentence = journal_text.split(".")[0][:50] if journal_text else "what you wrote"
        echo = first_sentence.strip() or "what you wrote"

    # Extract work object from journal text (with fallback)
    try:
        work_obj = extract_work_object(journal_text)
    except Exception:
        # Edge case 5: work object extraction fails
        work_obj = "task"

    # Look up the (emotion, behaviour) combination in the matrix
    key = (emotion, behaviour)
    entry = INTERVENTION_MATRIX.get(key)

    # Edge case 6: combination missing from matrix (should not happen with valid labels)
    if entry is None:
        return None

    # Format templates with echo phrase and work object
    try:
        reframe = entry["reframe"].format(echo=echo)
        task_text = entry["task_text"].format(work_obj=work_obj)
        why_this = entry["why_this"].format(work_obj=work_obj)
    except (KeyError, ValueError, IndexError):
        # Edge case 7: template formatting fails on unexpected characters
        return None

    # Identity belief is now woven into the reframe by personalise_reframe()
    # in the journal route (Task 2.7), so identity_echo is no longer needed here.

    return {
        "reframe": reframe,
        "micro_task": {
            "task_text": task_text,
            "estimated_minutes": entry.get("task_minutes", 2),
            "why_this": why_this,
        },
    }


def get_affirmation(predicted_emotion: str, user_id: int = None) -> str:
    """Return a personalised affirmation using the user's own identity belief if available."""

    # try to get the user's own identity belief from the database
    belief = None
    if user_id:
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT belief_text FROM identity_beliefs WHERE user_id = ? ORDER BY RANDOM() LIMIT 1",
                (user_id,),
            )
            row = cur.fetchone()
            if row:
                belief = row["belief_text"].strip()
        finally:
            conn.close()

    # if we have a belief, build a personalised affirmation around it
    if belief:
        personalised = {
            "overwhelmed": (
                "Everything feels like a lot right now, and that is okay. "
                "You said {belief}. Being here, writing this, is proof of that."
            ),
            "anxious": (
                "It makes sense that you feel anxious. That does not mean you cannot begin. "
                "You said {belief}. Showing up right now is what that person does."
            ),
            "stuck": (
                "Feeling stuck is not the same as being stuck. "
                "You said {belief}. You are here, and that is a step forward."
            ),
            "stressed": (
                "The pressure is real, and so is your ability to handle it at your own pace. "
                "You said {belief}. This moment is part of that."
            ),
            "tired": (
                "Rest is not giving up. It is part of the process. "
                "You said {belief}. Honouring your energy is something that person would do."
            ),
            "calm": (
                "This calm is something to notice. This is what it feels like when things settle. "
                "You said {belief}. Right now, you are living that."
            ),
            "frustrated": (
                "Frustration means you care about doing well. That energy is not wasted. "
                "You said {belief}. Channel this feeling into one small action."
            ),
            "guilty": (
                "Guilt means your values are still intact. That is not a bad thing. "
                "You said {belief}. One step now is all it takes to start closing the gap."
            ),
            "unmotivated": (
                "Not feeling motivated does not mean you have lost your way. "
                "You said {belief}. Showing up without motivation is the hardest kind of showing up, and it still counts."
            ),
            "hopeful": (
                "This hope is real. It is your brain noticing that things are shifting. "
                "You said {belief}. This feeling is evidence that the belief is taking hold."
            ),
            "proud": (
                "You earned this feeling. Hold on to it. "
                "You said {belief}. Today, you lived that."
            ),
        }
        template = personalised.get(
            predicted_emotion,
            "You said {belief}. Every time you show up like this, that belief gets a little more real."
        )
        return template.format(belief=belief)

    # fallback for users who have not set identity beliefs yet
    generic = {
        "overwhelmed": "I can take this slowly and still make progress.",
        "anxious": "I can steady myself and begin gently.",
        "stuck": "I move forward in small steps even when it feels hard.",
        "stressed": "I can pace myself and still get things done.",
        "tired": "I am allowed to pause without losing progress.",
        "calm": "I am becoming the version of myself I imagined.",
        "frustrated": "I can use this energy to move forward, not against myself.",
        "guilty": "I can start now. That is more powerful than guilt about yesterday.",
        "unmotivated": "I do not need motivation to take one small step.",
        "hopeful": "I trust this feeling. Things are starting to shift.",
        "proud": "I did something hard today. That is who I am becoming.",
    }
    return generic.get(predicted_emotion, "I am doing my best and that is enough.")


def get_alignment_state(user_id):
    # returns alignment score + streak for a specific user
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT alignment_score, emotional_streak, last_journal_date "
            "FROM alignment_state WHERE user_id = ?;",
            (user_id,),
        )
        row = cur.fetchone()
        if row is None:
            return 0, 0, None
        return row["alignment_score"], row["emotional_streak"], row["last_journal_date"]
    finally:
        conn.close()


def update_alignment_score(user_id, delta: int):
    # increase or reduce score but never below zero
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE alignment_state "
            "SET alignment_score = MAX(alignment_score + ?, 0) "
            "WHERE user_id = ?;",
            (delta, user_id),
        )
        conn.commit()
    finally:
        conn.close()


def update_emotional_streak_for_today(user_id):
    # journals on consecutive days then streak increases, skip a day resets
    today = date.today()
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT emotional_streak, last_journal_date FROM alignment_state WHERE user_id = ?;",
            (user_id,),
        )
        row = cur.fetchone()
        if row is None:
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
    finally:
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
            flash("Username needs to be at least 3 characters.", "error")
            return render_template("signup.html")

        if len(password) < 6:
            flash("Password needs to be at least 6 characters.", "error")
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
        return redirect(url_for('today'))

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
@onboarding_check
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

    cur.execute("SELECT id, goal_text FROM goals WHERE user_id = ? ORDER BY id ASC", (user_id,))
    goals = cur.fetchall()

    cur.execute("""
        SELECT ib.belief_text, g.goal_text
        FROM identity_beliefs ib
        LEFT JOIN goals g ON ib.linked_goal_id = g.id
        WHERE ib.user_id = ?
        ORDER BY ib.id ASC
    """, (user_id,))
    beliefs = cur.fetchall()

    cur.execute("SELECT COUNT(*) as count FROM journal_entries WHERE user_id = ?", (user_id,))
    journal_count = cur.fetchone()["count"]

    # Surface one random identity belief as today's reminder (RAS reprogramming)
    cur.execute(
        "SELECT belief_text FROM identity_beliefs WHERE user_id = ? ORDER BY RANDOM() LIMIT 1",
        (user_id,),
    )
    belief_row = cur.fetchone()
    today_belief = belief_row["belief_text"] if belief_row else None

    # Stage indicator: Awareness > Alignment > Action
    today_str_dash = date.today().isoformat()
    cur.execute(
        "SELECT COUNT(*) as count FROM habit_completions hc "
        "JOIN habits h ON hc.habit_id = h.id "
        "WHERE h.user_id = ? AND date(hc.completed_at) = ?",
        (user_id, today_str_dash),
    )
    habits_done_today = cur.fetchone()['count']
    cur.execute(
        "SELECT COUNT(*) as count FROM todos WHERE user_id = ? AND is_done = 1 AND date(created_at) = ?",
        (user_id, today_str_dash),
    )
    todos_done_today = cur.fetchone()['count']
    action_taken_today = (habits_done_today > 0 or todos_done_today > 0)
    has_beliefs = len(beliefs) > 0  # beliefs already fetched above

    if action_taken_today and has_beliefs and alignment_score > 0 and journal_count > 0:
        current_stage = 'action'
    elif has_beliefs and alignment_score > 0 and journal_count > 0:
        current_stage = 'alignment'
    elif journal_count > 0:
        current_stage = 'awareness'
    else:
        current_stage = None

    conn.close()

    today_affirmation = None
    if last_entry:
        _, emotion, _ = last_entry
        today_affirmation = get_affirmation(emotion, user_id)

    # Temporal analysis: detect emotion transitions across today's entries
    daily_insight = get_daily_insight(user_id)

    return render_template(
        "dashboard.html",
        greeting=get_greeting(),
        alignment_score=alignment_score,
        emotional_streak=emotional_streak,
        today_affirmation=today_affirmation,
        goals=goals,
        beliefs=beliefs,
        journal_count=journal_count,
        today_belief=today_belief,
        current_stage=current_stage,
        daily_insight=daily_insight,
    )

# main journaling page
@app.route("/journal", methods=["GET", "POST"])
@login_required
@onboarding_check
def journal():
    user_id = session['user_id']
    predicted_emotion = None
    predicted_behaviour = None
    identity_echo = None
    reframe = None
    affirmation = None
    micro_task = None
    new_entry_id = None
    paralysis_score = None
    paralysis_label = None
    paralysis_class = None
    daily_insight = None
    error_message = None

    if request.method == "POST":
        raw_text = request.form.get("entry_text", "")
        entry_text = normalise_entry_text(raw_text)

        # Edge case: empty or whitespace-only text submitted
        if not entry_text:
            error_message = "It looks like you did not write anything. That is okay. When you are ready, even one word is enough."
        # Edge case: extremely long text that could slow down ML models
        elif len(entry_text) > 5000:
            error_message = "That entry is quite long. Try keeping it under 5000 characters so the system can read it clearly."
            entry_text = None

        if entry_text:
            # Step 1: Run emotion model (Model 1)
            try:
                predicted_emotion = predict_emotion(entry_text)
            except Exception:
                # Edge case: emotion model fails on unexpected input
                predicted_emotion = None
                error_message = "Something went wrong while reading your entry. Please try again."

            # Step 2: Run behaviour model (Model 2)
            predicted_behaviour = predict_behaviour(entry_text)

            # Only continue full pipeline if emotion was detected
            if predicted_emotion:
                # Step 3: Select intervention (combined matrix or emotion-only fallback)
                intervention = get_combined_intervention(
                    entry_text, predicted_emotion, predicted_behaviour, user_id
                )

                if intervention:
                    # Combined intervention - both models working
                    reframe = intervention['reframe']
                    micro_task = intervention['micro_task']
                else:
                    # Fallback to emotion-only when behaviour model unavailable
                    reframe = generate_reframe(entry_text, predicted_emotion)
                    micro_task = generate_micro_task(entry_text, predicted_emotion)

                # Layer 3: Weave user's identity belief into the reframe (Task 2.7)
                reframe = personalise_reframe(
                    reframe, predicted_emotion, predicted_behaviour, user_id
                )

                # Step 4: Get personalised affirmation
                affirmation = get_affirmation(predicted_emotion, user_id)

                # Step 5: Calculate paralysis score
                paralysis_score = calculate_paralysis_score(
                    predicted_emotion, predicted_behaviour, entry_text, user_id
                )
                paralysis_label, paralysis_class = get_paralysis_label(paralysis_score)

                # Edge case: micro_task is None if both intervention paths fail
                if micro_task is None:
                    micro_task = {
                        "task_text": "Take one slow breath and notice how you feel.",
                        "estimated_minutes": 1,
                        "why_this": "When everything else is uncertain, your breath is something you can control.",
                    }

                # Step 6: Store all predictions in database
                conn = get_db_connection()
                try:
                    cur = conn.cursor()
                    cur.execute(
                        "INSERT INTO journal_entries "
                        "(user_id, entry_text, predicted_emotion, predicted_behaviour, "
                        "paralysis_score, reframe, micro_task_text, micro_task_minutes, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (user_id, entry_text, predicted_emotion, predicted_behaviour,
                         paralysis_score, reframe,
                         micro_task['task_text'], micro_task['estimated_minutes'],
                         datetime.now().isoformat(timespec="seconds")),
                    )
                    conn.commit()
                    new_entry_id = cur.lastrowid
                except Exception:
                    # Edge case: DB insert fails (locked DB, disk full, etc.)
                    error_message = "Your reflection was processed but could not be saved. Please try again."
                    predicted_emotion = None
                finally:
                    conn.close()

                # Only update scores and fetch insight if entry was saved
                if new_entry_id:
                    # journaling counts as identity-aligned behaviour
                    update_alignment_score(user_id, 1)
                    update_emotional_streak_for_today(user_id)

                    # Step 7: Check same-day context for temporal insight
                    # Called after save so the new entry is included
                    daily_insight = get_daily_insight(user_id)

    # load journal history
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, entry_text, predicted_emotion, predicted_behaviour, "
            "paralysis_score, reframe, micro_task_text, micro_task_minutes, created_at "
            "FROM journal_entries WHERE user_id = ? ORDER BY id DESC;",
            (user_id,),
        )
        entries = cur.fetchall()
    finally:
        conn.close()

    # Get streak for display
    _, emotional_streak, _ = get_alignment_state(user_id)

    # journal_count used to show first-use context on the page
    journal_count = len(entries)

    return render_template(
        "journal.html",
        predicted_emotion=predicted_emotion,
        predicted_behaviour=predicted_behaviour,
        identity_echo=identity_echo,
        reframe=reframe,
        affirmation=affirmation,
        micro_task=micro_task,
        new_entry_id=new_entry_id,
        paralysis_score=paralysis_score,
        paralysis_label=paralysis_label,
        paralysis_class=paralysis_class,
        entries=entries,
        emotional_streak=emotional_streak,
        journal_count=journal_count,
        daily_insight=daily_insight,
        error_message=error_message,
    )


@app.route("/journal/<int:entry_id>/edit", methods=["GET", "POST"])
@login_required
@onboarding_check
def edit_journal(entry_id):
    # edit an existing entry and re-predict new emotion
    user_id = session['user_id']

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, entry_text, predicted_emotion, reframe, micro_task_text, micro_task_minutes, created_at "
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

        # Edge case: empty text or text exceeding max length
        if entry_text and len(entry_text) <= 5000:
            # Step 1: Run emotion model
            try:
                new_emotion = predict_emotion(entry_text)
            except Exception:
                # Edge case: emotion model fails on unexpected input
                new_emotion = None

            # Step 2: Run behaviour model
            new_behaviour = predict_behaviour(entry_text)

            if new_emotion:
                # Step 3: Select intervention (combined or fallback)
                edit_intervention = get_combined_intervention(
                    entry_text, new_emotion, new_behaviour, user_id
                )
                if edit_intervention:
                    new_reframe = edit_intervention['reframe']
                    new_micro = edit_intervention['micro_task']
                else:
                    new_reframe = generate_reframe(entry_text, new_emotion)
                    new_micro = generate_micro_task(entry_text, new_emotion)

                # Layer 3: Weave user's identity belief into the edited reframe
                new_reframe = personalise_reframe(
                    new_reframe, new_emotion, new_behaviour, user_id
                )

                # Step 4: Recalculate paralysis score for edited entry
                new_paralysis = calculate_paralysis_score(
                    new_emotion, new_behaviour, entry_text, user_id
                )

                # Edge case: micro_task is None if both intervention paths fail
                if new_micro is None:
                    new_micro = {
                        "task_text": "Take one slow breath and notice how you feel.",
                        "estimated_minutes": 1,
                        "why_this": "When everything else is uncertain, your breath is something you can control.",
                    }

                # Step 5: Update entry in database
                try:
                    cur.execute(
                        "UPDATE journal_entries "
                        "SET entry_text = ?, predicted_emotion = ?, predicted_behaviour = ?, "
                        "paralysis_score = ?, reframe = ?, "
                        "micro_task_text = ?, micro_task_minutes = ? "
                        "WHERE id = ? AND user_id = ?;",
                        (entry_text, new_emotion, new_behaviour,
                         new_paralysis, new_reframe,
                         new_micro['task_text'], new_micro['estimated_minutes'],
                         entry_id, user_id),
                    )
                    conn.commit()
                except Exception:
                    # Edge case: DB update fails - redirect without changes
                    pass

        conn.close()
        return redirect(url_for("journal"))

    conn.close()
    return render_template("edit_journal.html", entry=entry)


@app.route("/journal/<int:entry_id>/delete", methods=["POST"])
@login_required
@onboarding_check
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


# --- Todo + Habit toggle routes ---

@app.route("/todo/add", methods=["POST"])
@login_required
def add_todo():
    user_id = session['user_id']
    text = request.form.get("text", "").strip()
    source = request.form.get("source", "manual")
    journal_entry_id = request.form.get("journal_entry_id")
    due_date = request.form.get("due_date")
    redirect_to = request.form.get("redirect", "today")

    if text:
        conn = get_db_connection()
        cur = conn.cursor()

        # Safely convert journal_entry_id to int (handle 'None' string from forms)
        journal_id = None
        if journal_entry_id and journal_entry_id != 'None':
            try:
                journal_id = int(journal_entry_id)
            except (ValueError, TypeError):
                journal_id = None

        cur.execute(
            "INSERT INTO todos (user_id, text, source, journal_entry_id, due_date, is_done, created_at) "
            "VALUES (?, ?, ?, ?, ?, 0, ?)",
            (user_id, text, source,
             journal_id,
             due_date if due_date else None,
             datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
        conn.close()

    if redirect_to == "journal":
        return redirect(url_for("journal"))
    return redirect(url_for("today"))


@app.route("/todo/<int:todo_id>/toggle", methods=["POST"])
@login_required
def toggle_todo(todo_id):
    user_id = session['user_id']
    conn = get_db_connection()
    cur = conn.cursor()

    # read current state before toggling so we know the direction
    cur.execute(
        "SELECT is_done FROM todos WHERE id = ? AND user_id = ?",
        (todo_id, user_id),
    )
    row = cur.fetchone()
    if row is None:
        conn.close()
        return redirect(url_for("today"))

    was_done = row["is_done"]

    cur.execute(
        "UPDATE todos SET is_done = CASE WHEN is_done = 0 THEN 1 ELSE 0 END "
        "WHERE id = ? AND user_id = ?",
        (todo_id, user_id),
    )
    conn.commit()
    conn.close()

    # only boost alignment when marking done, not when un-checking
    if not was_done:
        update_alignment_score(user_id, 1)

    redirect_to = request.form.get("redirect", "today")
    if redirect_to == "week":
        return redirect(url_for("week"))
    if not was_done:
        return redirect(url_for("today", todo_done="1"))
    return redirect(url_for("today"))


@app.route("/habit/<int:habit_id>/toggle", methods=["POST"])
@login_required
def toggle_habit(habit_id):
    user_id = session['user_id']
    today_str = date.today().isoformat()

    conn = get_db_connection()
    cur = conn.cursor()

    # verify habit belongs to user
    cur.execute("SELECT id FROM habits WHERE id = ? AND user_id = ?", (habit_id, user_id))
    if cur.fetchone() is None:
        conn.close()
        return redirect(url_for("today"))

    # check if already completed today
    cur.execute(
        "SELECT id FROM habit_completions WHERE habit_id = ? AND date(completed_at) = ?",
        (habit_id, today_str),
    )
    existing = cur.fetchone()

    if existing:
        # un-complete
        cur.execute("DELETE FROM habit_completions WHERE id = ?", (existing['id'],))
        conn.commit()
        conn.close()
        return redirect(url_for("today"))
    else:
        # mark complete
        cur.execute(
            "INSERT INTO habit_completions (habit_id, completed_at) VALUES (?, ?)",
            (habit_id, datetime.now().isoformat(timespec="seconds")),
        )
        update_alignment_score(user_id, 1)
        conn.commit()
        conn.close()
        return redirect(url_for("today", habit_done="1"))


@app.route("/todo/add-manual", methods=["POST"])
@login_required
def add_manual_todo():
    user_id = session['user_id']
    text = request.form.get("todo_text", "").strip()
    due_date = request.form.get("due_date")

    if text:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO todos (user_id, text, source, due_date, is_done, created_at) "
            "VALUES (?, ?, 'manual', ?, 0, ?)",
            (user_id, text, due_date if due_date else None,
             datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
        conn.close()

    return redirect(url_for("today"))


# --- Today + Week pages ---

@app.route("/today")
@login_required
@onboarding_check
def today():
    user_id = session['user_id']
    today_str = date.today().isoformat()

    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Today's todos: not done, and either no due date or due today or overdue
        cur.execute(
            "SELECT id, text, source, journal_entry_id, due_date, is_done, created_at "
            "FROM todos WHERE user_id = ? AND (is_done = 0 AND (due_date IS NULL OR due_date <= ?)) "
            "ORDER BY created_at DESC",
            (user_id, today_str),
        )
        active_todos = cur.fetchall()

        # Recently completed todos (done today)
        cur.execute(
            "SELECT id, text, source, is_done, created_at "
            "FROM todos WHERE user_id = ? AND is_done = 1 AND date(created_at) >= ? "
            "ORDER BY created_at DESC LIMIT 10",
            (user_id, today_str),
        )
        done_todos = cur.fetchall()

        # All active habits
        cur.execute(
            "SELECT id, name, is_sample FROM habits WHERE user_id = ? ORDER BY is_sample ASC, id ASC",
            (user_id,),
        )
        habits_rows = cur.fetchall()

        # Which habits are completed today
        cur.execute(
            "SELECT habit_id FROM habit_completions hc "
            "JOIN habits h ON hc.habit_id = h.id "
            "WHERE h.user_id = ? AND date(hc.completed_at) = ?",
            (user_id, today_str),
        )
        completed_habit_ids = {row['habit_id'] for row in cur.fetchall()}

        # Quick weekly overview: tasks due in next 7 days
        week_end = (date.today() + timedelta(days=7)).isoformat()
        cur.execute(
            "SELECT COUNT(*) as count FROM todos "
            "WHERE user_id = ? AND is_done = 0 AND due_date IS NOT NULL AND due_date BETWEEN ? AND ?",
            (user_id, today_str, week_end),
        )
        upcoming_count = cur.fetchone()['count']

        # Alignment data for greeting
        alignment_score, emotional_streak, _ = get_alignment_state(user_id)

        # Daily identity belief reminder (cycles through beliefs)
        cur.execute(
            "SELECT belief_text FROM identity_beliefs WHERE user_id = ? ORDER BY RANDOM() LIMIT 1",
            (user_id,),
        )
        belief_row = cur.fetchone()
        today_belief = belief_row["belief_text"] if belief_row else None

        # Weekly summary data
        week_start = (date.today() - timedelta(days=6)).isoformat()

        # Count active reflection days this week
        cur.execute(
            "SELECT COUNT(DISTINCT date(created_at)) as count "
            "FROM journal_entries WHERE user_id = ? AND date(created_at) BETWEEN ? AND ?",
            (user_id, week_start, today_str),
        )
        active_days_this_week = cur.fetchone()['count']

        # Count habits completed this week
        cur.execute(
            "SELECT COUNT(*) as count FROM habit_completions hc "
            "JOIN habits h ON hc.habit_id = h.id "
            "WHERE h.user_id = ? AND date(hc.completed_at) BETWEEN ? AND ?",
            (user_id, week_start, today_str),
        )
        habits_done_this_week = cur.fetchone()['count']

        # Week days indicator (last 7 days) - OPTIMIZED: Single query instead of 7
        cur.execute(
            "SELECT DISTINCT date(created_at) as entry_date "
            "FROM journal_entries WHERE user_id = ? AND date(created_at) BETWEEN ? AND ?",
            (user_id, week_start, today_str),
        )
        active_dates = {row['entry_date'] for row in cur.fetchall()}

        week_days = []
        day_names = ['M', 'T', 'W', 'T', 'F', 'S', 'S']
        day_full_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        for i in range(7):
            day_date = date.today() - timedelta(days=6-i)
            day_str = day_date.isoformat()

            week_days.append({
                'initial': day_names[day_date.weekday()],
                'name': day_full_names[day_date.weekday()],
                'active': day_str in active_dates,
                'is_today': day_str == today_str,
            })

        # Calculate stats for celebration
        total_habits = len(habits_rows)
        habits_completed_today = len(completed_habit_ids)
        all_habits_completed = (total_habits > 0 and habits_completed_today == total_habits)

        # Weekly encouragement message
        weekly_message = None
        if active_days_this_week >= 5:
            weekly_message = "You're showing up consistently. That's powerful."
        elif active_days_this_week >= 3:
            weekly_message = "You're building momentum, one day at a time."
        elif active_days_this_week >= 1:
            weekly_message = "You showed up this week. That counts."

        show_weekly_summary = (active_days_this_week > 0 or habits_done_this_week > 0)

        # Stage indicator: Awareness > Alignment > Action
        cur.execute("SELECT COUNT(*) as count FROM journal_entries WHERE user_id = ?", (user_id,))
        journal_count = cur.fetchone()['count']
        cur.execute("SELECT COUNT(*) as count FROM identity_beliefs WHERE user_id = ?", (user_id,))
        has_beliefs = cur.fetchone()['count'] > 0
        action_taken_today = (habits_completed_today > 0 or len(done_todos) > 0)

        if action_taken_today and has_beliefs and alignment_score > 0 and journal_count > 0:
            current_stage = 'action'
        elif has_beliefs and alignment_score > 0 and journal_count > 0:
            current_stage = 'alignment'
        elif journal_count > 0:
            current_stage = 'awareness'
        else:
            current_stage = None

    finally:
        conn.close()

    todo_done = request.args.get("todo_done") == "1"
    habit_done = request.args.get("habit_done") == "1"

    return render_template(
        "today.html",
        greeting=get_greeting(),
        active_todos=active_todos,
        done_todos=done_todos,
        habits=habits_rows,
        completed_habit_ids=completed_habit_ids,
        upcoming_count=upcoming_count,
        alignment_score=alignment_score,
        emotional_streak=emotional_streak,
        today_str=today_str,
        # Weekly summary data
        show_weekly_summary=show_weekly_summary,
        week_days=week_days,
        active_days_this_week=active_days_this_week,
        habits_done_this_week=habits_done_this_week,
        weekly_message=weekly_message,
        # Habit completion stats
        total_habits=total_habits,
        habits_completed_today=habits_completed_today,
        all_habits_completed=all_habits_completed,
        today_belief=today_belief,
        # Win acknowledgment flags
        todo_done=todo_done,
        habit_done=habit_done,
        current_stage=current_stage,
    )


@app.route("/week")
@login_required
@onboarding_check
def week():
    user_id = session['user_id']
    today_obj = date.today()
    today_str = today_obj.isoformat()
    week_end = (today_obj + timedelta(days=7)).isoformat()
    week_start = (today_obj - timedelta(days=6)).isoformat()

    conn = get_db_connection()
    cur = conn.cursor()

    # Tasks due this week (next 7 days)
    cur.execute(
        "SELECT id, text, due_date, is_done, created_at "
        "FROM todos WHERE user_id = ? AND due_date IS NOT NULL AND due_date BETWEEN ? AND ? "
        "ORDER BY due_date ASC, created_at ASC",
        (user_id, today_str, week_end),
    )
    week_todos = cur.fetchall()

    # All tasks not done with no due date
    cur.execute(
        "SELECT id, text, due_date, is_done, created_at "
        "FROM todos WHERE user_id = ? AND is_done = 0 AND due_date IS NULL "
        "ORDER BY created_at DESC LIMIT 15",
        (user_id,),
    )
    undated_todos = cur.fetchall()

    # Habit completion summary for last 7 days
    cur.execute(
        "SELECT h.id, h.name, COUNT(hc.id) as completions "
        "FROM habits h "
        "LEFT JOIN habit_completions hc ON h.id = hc.habit_id AND date(hc.completed_at) >= ? "
        "WHERE h.user_id = ? "
        "GROUP BY h.id ORDER BY h.id ASC",
        (week_start, user_id),
    )
    habit_summary = cur.fetchall()

    # Journal count this week
    cur.execute(
        "SELECT COUNT(*) as count FROM journal_entries "
        "WHERE user_id = ? AND date(created_at) >= ?",
        (user_id, week_start),
    )
    journal_count = cur.fetchone()['count']

    conn.close()

    return render_template(
        "week.html",
        week_todos=week_todos,
        undated_todos=undated_todos,
        habit_summary=habit_summary,
        journal_count=journal_count,
        today_str=today_str,
    )


@app.route("/habits", methods=["GET", "POST"])
@login_required
@onboarding_check
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
            return redirect(url_for("habits", acknowledged="1"))

    # show all habits for this user
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT h.id, h.name, h.is_sample, h.created_at, g.goal_text
        FROM habits h
        LEFT JOIN goals g ON h.linked_goal_id = g.id
        WHERE h.user_id = ?
        ORDER BY h.id ASC
    """, (user_id,))
    habits_rows = cur.fetchall()
    conn.close()

    acknowledged = request.args.get("acknowledged") == "1"
    return render_template("habits.html", habits=habits_rows, acknowledged=acknowledged)

@app.route("/analytics")
@login_required
@onboarding_check
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


# ====================================
# ONBOARDING ROUTES
# ====================================

@app.route("/onboarding/goals", methods=["GET", "POST"])
@login_required
def onboarding_goals():
    """Step 1: Enter top 3 goals for next 3 months"""
    user_id = session['user_id']

    if request.method == "POST":
        goal_1 = request.form.get("goal_1", "").strip()
        goal_2 = request.form.get("goal_2", "").strip()
        goal_3 = request.form.get("goal_3", "").strip()

        if not goal_1 or not goal_2 or not goal_3:
            flash("Please enter all 3 goals.", "error")
            return render_template("onboarding_goals.html", step=1, total_steps=4)

        # Delete existing goals if any (allow re-doing this step)
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM goals WHERE user_id = ?", (user_id,))

        # Insert new goals
        now = datetime.now().isoformat(timespec="seconds")
        for goal_text in [goal_1, goal_2, goal_3]:
            cur.execute(
                "INSERT INTO goals (user_id, goal_text, created_at) VALUES (?, ?, ?)",
                (user_id, goal_text, now)
            )
        conn.commit()
        conn.close()

        return redirect(url_for('onboarding_beliefs'))

    # GET: load existing goals if any
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT goal_text FROM goals WHERE user_id = ? ORDER BY id ASC", (user_id,))
    existing_goals = [row['goal_text'] for row in cur.fetchall()]
    conn.close()

    return render_template(
        "onboarding_goals.html",
        step=1,
        total_steps=4,
        existing_goals=existing_goals if existing_goals else ["", "", ""]
    )


@app.route("/onboarding/beliefs", methods=["GET", "POST"])
@login_required
def onboarding_beliefs():
    """Step 2: Rewrite goals as identity beliefs"""
    user_id = session['user_id']

    # Load user's goals
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, goal_text FROM goals WHERE user_id = ? ORDER BY id ASC", (user_id,))
    goals = cur.fetchall()
    conn.close()

    if not goals:
        flash("Please complete the goals step first.", "error")
        return redirect(url_for('onboarding_goals'))

    if request.method == "POST":
        beliefs_data = []
        for goal in goals:
            belief_key = f"belief_{goal['id']}"
            belief_text = request.form.get(belief_key, "").strip()
            if belief_text:
                beliefs_data.append((goal['id'], belief_text))

        if len(beliefs_data) < len(goals):
            flash("Please create an identity belief for each goal.", "error")
            return render_template("onboarding_beliefs.html", step=2, total_steps=4, goals=goals)

        # Delete existing beliefs and insert new ones
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM identity_beliefs WHERE user_id = ?", (user_id,))

        now = datetime.now().isoformat(timespec="seconds")
        for goal_id, belief_text in beliefs_data:
            cur.execute(
                "INSERT INTO identity_beliefs (user_id, belief_text, linked_goal_id, created_at) "
                "VALUES (?, ?, ?, ?)",
                (user_id, belief_text, goal_id, now)
            )
        conn.commit()
        conn.close()

        return redirect(url_for('onboarding_thoughts'))

    # GET: load existing beliefs if any
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT linked_goal_id, belief_text FROM identity_beliefs WHERE user_id = ?",
        (user_id,)
    )
    existing_beliefs = {row['linked_goal_id']: row['belief_text'] for row in cur.fetchall()}
    conn.close()

    return render_template(
        "onboarding_beliefs.html",
        step=2,
        total_steps=4,
        goals=goals,
        existing_beliefs=existing_beliefs
    )


@app.route("/onboarding/thoughts", methods=["GET", "POST"])
@login_required
def onboarding_thoughts():
    """Step 3: Add positive thoughts linked to beliefs"""
    user_id = session['user_id']

    # Load user's beliefs
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, belief_text FROM identity_beliefs WHERE user_id = ? ORDER BY id ASC",
        (user_id,)
    )
    beliefs = cur.fetchall()
    conn.close()

    if not beliefs:
        flash("Please complete the identity beliefs step first.", "error")
        return redirect(url_for('onboarding_beliefs'))

    if request.method == "POST":
        thoughts_data = []
        for belief in beliefs:
            thought_key = f"thought_{belief['id']}"
            thought_text = request.form.get(thought_key, "").strip()
            if thought_text:
                thoughts_data.append((belief['id'], thought_text))

        if len(thoughts_data) < len(beliefs):
            flash("Please add at least one positive thought for each belief.", "error")
            return render_template("onboarding_thoughts.html", step=3, total_steps=4, beliefs=beliefs)

        # Delete existing thoughts and insert new ones
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM positive_thoughts WHERE user_id = ?", (user_id,))

        now = datetime.now().isoformat(timespec="seconds")
        for belief_id, thought_text in thoughts_data:
            cur.execute(
                "INSERT INTO positive_thoughts (user_id, thought_text, linked_belief_id, created_at) "
                "VALUES (?, ?, ?, ?)",
                (user_id, thought_text, belief_id, now)
            )
        conn.commit()
        conn.close()

        return redirect(url_for('onboarding_habits'))

    # GET: load existing thoughts if any
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT linked_belief_id, thought_text FROM positive_thoughts WHERE user_id = ?",
        (user_id,)
    )
    existing_thoughts = {row['linked_belief_id']: row['thought_text'] for row in cur.fetchall()}
    conn.close()

    return render_template(
        "onboarding_thoughts.html",
        step=3,
        total_steps=4,
        beliefs=beliefs,
        existing_thoughts=existing_thoughts
    )


@app.route("/onboarding/habits", methods=["GET", "POST"])
@login_required
def onboarding_habits():
    """Step 4: Create small daily habits linked to goals"""
    user_id = session['user_id']

    # Load user's goals
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, goal_text FROM goals WHERE user_id = ? ORDER BY id ASC", (user_id,))
    goals = cur.fetchall()
    conn.close()

    if not goals:
        flash("Please complete the goals step first.", "error")
        return redirect(url_for('onboarding_goals'))

    if request.method == "POST":
        habits_data = []
        for goal in goals:
            habit_key = f"habit_{goal['id']}"
            habit_text = request.form.get(habit_key, "").strip()
            if habit_text:
                habits_data.append((goal['id'], habit_text))

        if len(habits_data) < len(goals):
            flash("Please create at least one habit for each goal.", "error")
            return render_template("onboarding_habits.html", step=4, total_steps=4, goals=goals)

        # Delete existing custom habits (is_sample = 0) and insert new ones
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM habits WHERE user_id = ? AND is_sample = 0", (user_id,))

        now = datetime.now().isoformat(timespec="seconds")
        for goal_id, habit_text in habits_data:
            cur.execute(
                "INSERT INTO habits (user_id, name, is_sample, linked_goal_id, created_at) "
                "VALUES (?, ?, 0, ?, ?)",
                (user_id, habit_text, goal_id, now)
            )

        # Mark onboarding as complete
        cur.execute("UPDATE users SET onboarding_complete = 1 WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()

        flash("Onboarding complete! Welcome to your journey.", "success")
        return redirect(url_for('today'))

    # GET: load existing custom habits if any
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT linked_goal_id, name FROM habits WHERE user_id = ? AND is_sample = 0",
        (user_id,)
    )
    existing_habits = {row['linked_goal_id']: row['name'] for row in cur.fetchall()}
    conn.close()

    return render_template(
        "onboarding_habits.html",
        step=4,
        total_steps=4,
        goals=goals,
        existing_habits=existing_habits
    )


if __name__ == "__main__":
    init_db()
    migrate_db_for_onboarding()
    app.run(debug=True)
