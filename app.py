from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import joblib
import os
from datetime import datetime, date, timedelta

# creates flask app instance
app = Flask(__name__)

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

    # journal entries table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS journal_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_text TEXT NOT NULL,
            predicted_emotion TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)

    # alignment score + emotional streak (single row with id = 1)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS alignment_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            alignment_score INTEGER NOT NULL,
            emotional_streak INTEGER NOT NULL,
            last_journal_date TEXT
        );
    """)

    # habit list table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS habits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            is_sample INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
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

     # create starting values if empty
    cur.execute("SELECT COUNT(*) AS cnt FROM alignment_state;")
    row = cur.fetchone()
    if row["cnt"] == 0:
        cur.execute(
            "INSERT INTO alignment_state (id, alignment_score, emotional_streak, last_journal_date) "
            "VALUES (1, 0, 0, NULL);"
        )

    # add a few sample habits if none exist yet
    cur.execute("SELECT COUNT(*) AS cnt FROM habits;")
    row = cur.fetchone()
    if row["cnt"] == 0:
        now = datetime.now().isoformat(timespec="seconds")
        sample_habits = [
            "Write one sentence for an assignment",
            "Open my notes and read for 5 minutes",
            "Tidy my desk for two minutes",
        ]
        for name in sample_habits:
            cur.execute(
                "INSERT INTO habits (name, is_sample, created_at) VALUES (?, ?, ?)",
                (name, 1, now),
            )

    conn.commit()
    conn.close()


# support functions

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
            "Feeling stuck doesn’t mean you can’t do it. "
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


def get_alignment_state():
    # returns alignment score + streak for dashboard
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT alignment_score, emotional_streak, last_journal_date "
        "FROM alignment_state WHERE id = 1;"
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return 0, 0, None
    return row["alignment_score"], row["emotional_streak"], row["last_journal_date"]


def update_alignment_score(delta: int):
    # increase or reduce score but never below zero
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE alignment_state "
        "SET alignment_score = MAX(alignment_score + ?, 0) "
        "WHERE id = 1;",
        (delta,),
    )
    conn.commit()
    conn.close()


def update_emotional_streak_for_today():
    # journals on consecutive days then streak increases, skip a day resets
    today = date.today()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT emotional_streak, last_journal_date FROM alignment_state WHERE id = 1;"
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
        "WHERE id = 1;",
        (streak, today.isoformat()),
    )
    conn.commit()
    conn.close()


# routes - UI screen

@app.route("/")
def dashboard():
    # homepage showing score + streak + affirmation
    alignment_score, emotional_streak, last_journal_date = get_alignment_state()

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT entry_text, predicted_emotion, created_at "
        "FROM journal_entries ORDER BY id DESC LIMIT 1;"
    )
    last_entry = cur.fetchone()
    conn.close()

    today_affirmation = None
    if last_entry:
        _, emotion, _ = last_entry
        _, today_affirmation = get_reframe_and_affirmation(emotion)

    return render_template(
        "dashboard.html",
        alignment_score=alignment_score,
        emotional_streak=emotional_streak,
        today_affirmation=today_affirmation,
    )

# main journaling page
@app.route("/journal", methods=["GET", "POST"])
def journal():
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
                "INSERT INTO journal_entries (entry_text, predicted_emotion, created_at) "
                "VALUES (?, ?, ?)",
                (entry_text, predicted_emotion,
                 datetime.now().isoformat(timespec="seconds")),
            )
            conn.commit()
            conn.close()

            # journaling counts as identity-aligned behaviour
            update_alignment_score(1)
            update_emotional_streak_for_today()

    # load journal history
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, entry_text, predicted_emotion, created_at "
        "FROM journal_entries ORDER BY id DESC;"
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
def edit_journal(entry_id):
    # edit an existing entry and re-predict new emotion

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, entry_text, predicted_emotion, created_at "
        "FROM journal_entries WHERE id = ?;",
        (entry_id,),
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
                "WHERE id = ?;",
                (entry_text, new_emotion, entry_id),
            )
            conn.commit()

        conn.close()
        return redirect(url_for("journal"))

    conn.close()
    return render_template("edit_journal.html", entry=entry)


@app.route("/journal/<int:entry_id>/delete", methods=["POST"])
def delete_journal(entry_id):
    # delete a journal entry and gently reduce alignment score by 1

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM journal_entries WHERE id = ?;", (entry_id,))
    conn.commit()
    conn.close()

    # deleting an entry removes one point but never below zero
    update_alignment_score(-1)

    return redirect(url_for("journal"))


@app.route("/habits", methods=["GET", "POST"])
def habits():
    """
    habits page:
    - show sample + user-added habits
    - allow ticking habits as completed
    - allow adding new habits/tasks
    """

    if request.method == "POST":
        # if user added a new habit/task
        new_habit = request.form.get("new_habit", "").strip()
        if new_habit:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO habits (name, is_sample, created_at) "
                "VALUES (?, ?, ?);",
                (new_habit, 0, datetime.now().isoformat(timespec="seconds")),
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

                cur.execute(
                    "INSERT INTO habit_completions (habit_id, completed_at) "
                    "VALUES (?, ?);",
                    (hid_int, now),
                )
                # each completed habit counts as identity-aligned behaviour
                update_alignment_score(1)

            conn.commit()
            conn.close()
            return redirect(url_for("habits"))

    # show all habits
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, is_sample, created_at "
        "FROM habits ORDER BY id ASC;"
    )
    habits_rows = cur.fetchall()
    conn.close()

    return render_template("habits.html", habits=habits_rows)

@app.route("/analytics")
def analytics():
    conn = get_db_connection()
    cur = conn.cursor()

    # 1. emotion distribution across all journal entries
    cur.execute("""
        SELECT predicted_emotion, COUNT(*) as count
        FROM journal_entries
        GROUP BY predicted_emotion
    """)
    emotion_data = cur.fetchall()

    # 2. days user journaled in last 7 days
    today = date.today()
    last_week = (today - timedelta(days=6)).isoformat()
    cur.execute("""
        SELECT COUNT(DISTINCT date(created_at)) AS active_days
        FROM journal_entries
        WHERE date(created_at) >= ?
    """, (last_week,))
    active_days = cur.fetchone()["active_days"]

    # 3. habit completions in last 7 days
    cur.execute("""
        SELECT COUNT(*) AS habits_done
        FROM habit_completions
        WHERE date(completed_at) >= ?
    """, (last_week,))
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
