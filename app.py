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
            f'The question "{echo}" is a trap — it demands certainty about something unknowable, which keeps you spinning without moving. Anxiety mistakes repetition for preparation. What you need isn\'t an answer to the question; you need to stop asking it. Pick one action that exists outside the loop — send a message, open a file, write one sentence — and watch the urgency lose its grip.',

            # 1: avoidance cycle (open/close)
            f'"{echo}" is threshold fear — the moment before starting feels unbearable, so you close the distance and retreat. But the fear isn\'t about the work; it\'s about crossing from safe to uncertain. The work itself is neutral. Open the file and leave it visible for 5 minutes without touching it. Let proximity replace avoidance as the new normal.',

            # 2: catastrophizing
            f'The terror that "{echo}" is your mind confusing *feeling* in danger with *being* in danger. Fear amplifies the stakes to justify its own intensity, but the intensity isn\'t evidence. The task won\'t collapse you. Write one rough sentence with the explicit goal of it being bad. Show yourself the gap between dread and reality.',

            # 3: social fear
            f'You\'re trapped between your inner experience and what you imagine others see — and that gap feels like proof something is wrong. But you\'re comparing your fear to their façade. They can\'t see your panic, and you can\'t see their doubt. Do one external action (open your notes, type one line) to anchor yourself in what\'s real, not what you think they think.',

            # 4: general anxious
            f'Worry is rehearsal disguised as protection — it creates the illusion of control over outcomes you can\'t influence yet. But spinning doesn\'t prevent disaster; it just exhausts you before anything happens. The way out isn\'t calm; it\'s movement. Pick the smallest concrete step available right now and take it. Let action replace rumination as the thing you do when uncertain.',
        ]
        return templates[template_idx]

    # === OVERWHELMED TEMPLATES ===
    elif emotion_label == "overwhelmed":
        templates = [
            # 0: paralysis from choice
            f'"{echo}" is decision paralysis — you\'re frozen because choosing one task feels like abandoning the others. But the paralysis itself is what\'s failing them all. There is no optimal sequence; there\'s only the cost of waiting for one to appear. Choose by deadline, by ease, or at random. Forward motion makes the next choice clearer. Pick one now.',

            # 1: volume overload
            f'The sheer amount — "{echo}" — is creating the illusion that you need to hold it all in your mind at once. You don\'t. Overwhelm is a capacity error: trying to process everything simultaneously instead of sequentially. Release everything but one. Do that one thing for 10 minutes as if nothing else exists. Let the rest wait in the dark.',

            # 2: holding all at once
            f'You\'re experiencing "{echo}" because you\'ve collapsed separate tasks into one crushing mass. Overwhelm happens when the boundaries between things dissolve. The antidote is artificial narrowing: write down three tasks, close the list, and do only the first one for 15 minutes. Shrink your world until it\'s survivable, then act inside it.',

            # 3: general overwhelmed
            f'Everything feels urgent because urgency is the emotion of overload, not a fact about your tasks. When everything screams for attention, none of them are actually louder — you just can\'t tell the difference. The way forward is arbitrary choice. Pick the task that would quiet the loudest inner voice, or the one you could finish fastest. Do that one badly. Let completion replace perfection.',
        ]
        return templates[template_idx]

    # === STRESSED TEMPLATES ===
    elif emotion_label == "stressed":
        templates = [
            # 0: time urgency
            f'"{echo}" is your body interpreting time as a closing fist — the scarcity creates tunnel vision that makes solutions invisible. But deadline pressure doesn\'t require perfection; it requires output. You don\'t need to feel capable; you need to move. Set a 25-minute boundary and produce something rough. Speed, not quality. You can refine it later when the fist opens.',

            # 1: pressure reducing thinking
            f'The stress you describe — "{echo}" — is narrowing your thinking into a defensive crouch, which makes everything harder to see. Pressure doesn\'t sharpen focus; it collapses it. But you don\'t need expanded thinking right now; you need reduced scope. Choose the most obvious next action and do only that for 20 minutes. Let momentum replace clarity as the engine.',

            # 2: stress blocking cognition
            f'Stress is convincing you that confusion means incapacity — that because you can\'t think clearly, you can\'t act at all. That\'s backward. Clarity is the result of action, not the prerequisite. Your hands don\'t need your mind\'s permission. Open the file and type for 10 minutes with zero expectation of coherence. Let motion generate the clarity stress is withholding.',

            # 3: general stressed
            f'Urgency is demanding immediate perfection, which is impossible and paralyzing. But stress distorts time — what feels like "everything now" is actually "something next, then something after." You don\'t need the whole task; you need the next 15 minutes. Do the most obvious fragment badly. Survival output is still output. Polish is a luxury for after the deadline passes.',
        ]
        return templates[template_idx]

    # === AVOIDANT TEMPLATES ===
    elif emotion_label == "avoidant":
        templates = [
            # 0: start-stop cycle
            f'The pattern "{echo}" is a conditioning loop — every retreat teaches the threat response to fire faster next time. Avoidance feels like self-protection but functions as self-training: the task becomes more dangerous with each escape. The loop breaks through exposure without completion. Open the file and sit with it for 5 minutes. Don\'t work. Just stop running. Teach yourself that proximity isn\'t danger.',

            # 1: explicit avoidance
            f'"{echo}" is the ache of knowing and not doing — and that gap is filled with shame disguised as relief. But avoidance doesn\'t reduce the discomfort; it compounds it with each delay. The task isn\'t getting easier by waiting; it\'s accumulating dread. Do the smallest possible fragment now — one sentence, one line, one click — without the burden of finishing. Just interrupt the pattern of retreat.',

            # 2: postponing
            f'The bargain "{echo}" is your mind trading present discomfort for the fantasy of future readiness. But later you will feel exactly what you feel now, plus the weight of continued delay. Motivation is the reward for starting, not the prerequisite. Set a 2-minute timer and do a miniature version of the task. Not well. Not completely. Just immediately. Two minutes to prove later is a lie.',

            # 3: distraction
            f'"{echo}" is the relief mechanism in real time — the discomfort of starting sends you toward anything softer, and the pattern strengthens each time you obey it. But distraction is debt: the task waits, and the next approach will be harder. Reverse the incentive. Do one small action on the task *first* — open one file, write one phrase — and earn the distraction after. Make avoidance cost something.',

            # 4: general avoidant
            f'You know what to do, but knowing isn\'t the obstacle — it\'s the absence of a feeling you\'re waiting for that will never arrive. Readiness, motivation, clarity: none of them precede action. They follow it. You don\'t need permission from your emotions to begin. Start in whatever state you\'re in — uncertain, resistant, afraid. Write one bad sentence. Let action create the feeling, not the reverse.',
        ]
        return templates[template_idx]

    # === DISCOURAGED TEMPLATES ===
    elif emotion_label == "discouraged":
        templates = [
            # 0: hopelessness
            f'When you think "{echo}", you\'re collapsing effort and outcome into the same thing — but they\'re not. Effort is yours; outcome is circumstance. Hopelessness mistakes uncertainty for inevitability. You can\'t see the result yet, but that doesn\'t mean trying is empty. It means trying is the only honest response to not knowing. Do 5 minutes of work as an experiment, not a solution. Test whether effort still holds meaning when success isn\'t promised.',

            # 1: absolutist thinking
            f'"{echo}" is the language of all-or-nothing — and absolutes erase the middle where most of living happens. You\'re frozen in a binary that doesn\'t exist: always failing or never capable. But patterns aren\'t permanent, and one moment doesn\'t predict all future ones. The story you\'re telling collapses time. Do one thing differently today — even trivially small — to prove the absolute is breakable. Let the exception shatter the rule.',

            # 2: low self-belief
            f'The belief "{echo}" is treating a current limitation as a fixed identity. But what you can\'t do yet isn\'t the same as what you are. The gap between where you are and where you want to be isn\'t evidence of inadequacy — it\'s the space where capability grows. You only see the gap because you can imagine better. Do one small imperfect thing today. Not to prove you\'re capable. To prove you can still try when you don\'t believe you are.',

            # 3: general discouraged
            f'You\'re reading the difficulty as a verdict — if it\'s hard, you must not be built for it. But struggle is information about the learning curve, not your ceiling. The absence of ease doesn\'t mean the presence of impossibility. Do the smallest version of the task available — one sentence, one step, one attempt — not to succeed, but to stay in contact with forward motion. Let trying be the point when winning feels out of reach.',
        ]
        return templates[template_idx]

    # === CALM TEMPLATES ===
    elif emotion_label == "calm":
        templates = [
            # 0: clarity
            f'The clarity you describe — "{echo}" — is rare and worth protecting, but not by freezing or forcing productivity. Calm isn\'t a resource to extract; it\'s a state to inhabit. Take one gentle action from this place, not because you need to justify the feeling, but because ease is what makes sustainable work possible. Let the calm inform the pace, not the urgency.',

            # 1: general calm
            f'This steadiness is valuable precisely because it asks nothing of you. You don\'t need to perform with it or amplify it into hyperproductivity. Just notice what it feels like to not be fighting yourself. If something wants to be done, do it lightly. If nothing does, rest here. Let calm be its own destination, not a launching pad.',
        ]
        return templates[template_idx]

    # === NEUTRAL / OTHER ===
    else:
        return f'You wrote this down, which means part of you is still reaching toward change even when the path isn\'t clear. Awareness is the beginning of movement. You don\'t need a map or a plan — just the next smallest step. Pick something near, something easy, or something unfinished. Do it for 10 minutes. Let action clarify what reflection can\'t.'


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

    if emotion_label == "stuck":
        emotion_label = "avoidant"

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
            'task_text': f"{actions['open']} and leave it visible for 2 minutes. Don't type — just sit with it open.",
            'estimated_minutes': 2,
            'why_this': f"Anxiety spikes at the threshold of starting. Having your {work_obj} open without pressure reduces that spike over time.",
        }
    elif emotion_label == "overwhelmed":
        return {
            'task_text': f"{actions['small']}. Just that one thing — ignore everything else for now.",
            'estimated_minutes': 1,
            'why_this': f"Overwhelm dissolves when you shrink scope to a single visible action on your {work_obj}.",
        }
    elif emotion_label == "stressed":
        return {
            'task_text': f"{actions['write']} — messy and imperfect. Speed over quality for 2 minutes.",
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
            'task_text': f"{actions['write']} — not to finish, just to prove movement is still possible.",
            'estimated_minutes': 1,
            'why_this': f"Discouragement says effort is pointless. One sentence in your {work_obj} says otherwise.",
        }
    elif emotion_label == "calm":
        return {
            'task_text': f"{actions['review']} and note one thing you want to work on next.",
            'estimated_minutes': 2,
            'why_this': f"Calm is the best state to gently re-engage with your {work_obj} without pressure.",
        }
    else:
        return {
            'task_text': f"{actions['open']} and spend 2 minutes looking at where you left off.",
            'estimated_minutes': 2,
            'why_this': f"Re-engaging with your {work_obj} for 2 minutes builds momentum without pressure.",
        }


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
    # public landing page — redirect to today if already logged in
    if 'user_id' in session:
        return redirect(url_for('today'))
    return render_template("landing.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if 'user_id' in session:
        return redirect(url_for('today'))

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
        return redirect(url_for('today'))

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
            return redirect(url_for('today'))

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
        goals=goals,
        beliefs=beliefs,
        journal_count=journal_count,
    )

# main journaling page
@app.route("/journal", methods=["GET", "POST"])
@login_required
@onboarding_check
def journal():
    user_id = session['user_id']
    predicted_emotion = None
    reframe = None
    affirmation = None
    micro_task = None
    new_entry_id = None

    if request.method == "POST":
        raw_text = request.form.get("entry_text", "")
        entry_text = normalise_entry_text(raw_text)

        if entry_text:
            predicted_emotion = predict_emotion(entry_text)
            reframe = generate_reframe(entry_text, predicted_emotion)
            affirmation = get_affirmation(predicted_emotion)
            micro_task = generate_micro_task(entry_text, predicted_emotion)

            conn = get_db_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO journal_entries "
                    "(user_id, entry_text, predicted_emotion, reframe, micro_task_text, micro_task_minutes, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (user_id, entry_text, predicted_emotion, reframe,
                     micro_task['task_text'], micro_task['estimated_minutes'],
                     datetime.now().isoformat(timespec="seconds")),
                )
                conn.commit()
                new_entry_id = cur.lastrowid
            finally:
                conn.close()

            # journaling counts as identity-aligned behaviour
            update_alignment_score(user_id, 1)
            update_emotional_streak_for_today(user_id)

    # load journal history
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, entry_text, predicted_emotion, reframe, micro_task_text, micro_task_minutes, created_at "
            "FROM journal_entries WHERE user_id = ? ORDER BY id DESC;",
            (user_id,),
        )
        entries = cur.fetchall()
    finally:
        conn.close()

    # Get streak for display
    _, emotional_streak, _ = get_alignment_state(user_id)

    return render_template(
        "journal.html",
        predicted_emotion=predicted_emotion,
        reframe=reframe,
        affirmation=affirmation,
        micro_task=micro_task,
        new_entry_id=new_entry_id,
        entries=entries,
        emotional_streak=emotional_streak,
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

        if entry_text:
            new_emotion = predict_emotion(entry_text)
            new_reframe = generate_reframe(entry_text, new_emotion)
            new_micro = generate_micro_task(entry_text, new_emotion)
            cur.execute(
                "UPDATE journal_entries "
                "SET entry_text = ?, predicted_emotion = ?, reframe = ?, "
                "micro_task_text = ?, micro_task_minutes = ? "
                "WHERE id = ? AND user_id = ?;",
                (entry_text, new_emotion, new_reframe,
                 new_micro['task_text'], new_micro['estimated_minutes'],
                 entry_id, user_id),
            )
            conn.commit()

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
    cur.execute(
        "UPDATE todos SET is_done = CASE WHEN is_done = 0 THEN 1 ELSE 0 END "
        "WHERE id = ? AND user_id = ?",
        (todo_id, user_id),
    )
    conn.commit()
    conn.close()

    # completing a todo boosts alignment
    update_alignment_score(user_id, 1)

    redirect_to = request.form.get("redirect", "today")
    if redirect_to == "week":
        return redirect(url_for("week"))
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
    else:
        # mark complete
        cur.execute(
            "INSERT INTO habit_completions (habit_id, completed_at) VALUES (?, ?)",
            (habit_id, datetime.now().isoformat(timespec="seconds")),
        )
        update_alignment_score(user_id, 1)

    conn.commit()
    conn.close()
    return redirect(url_for("today"))


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
            weekly_message = "You're showing up consistently — that's powerful."
        elif active_days_this_week >= 3:
            weekly_message = "You're building momentum, one day at a time."
        elif active_days_this_week >= 1:
            weekly_message = "You showed up this week. That counts."

        show_weekly_summary = (active_days_this_week > 0 or habits_done_this_week > 0)

    finally:
        conn.close()

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
            return redirect(url_for("habits"))

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

    return render_template("habits.html", habits=habits_rows)

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
