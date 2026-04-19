# create_test_user2.py
# populates supabase postgres with a realistic demo user for FYP screenshots
# run with: python create_test_user2.py from the project root

import os
import sys
import psycopg2
import psycopg2.extras
import joblib
from datetime import datetime, date, timedelta
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv


# ---- config ----
USERNAME = "demo_user"
PASSWORD = "DemoUser2026!"


# ---- load environment ----
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    print("ERROR: No DATABASE_URL found in .env file.")
    sys.exit(1)


# ---- load ML models ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

print("Loading ML models...")
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
emotion_model = joblib.load(os.path.join(MODEL_DIR, "emotion_model.pkl"))
behaviour_vectorizer = joblib.load(os.path.join(MODEL_DIR, "behaviour_vectorizer.pkl"))
behaviour_model = joblib.load(os.path.join(MODEL_DIR, "behaviour_model.pkl"))
print("  Emotion model loaded (11 classes)")
print("  Behaviour model loaded (6 classes)")


def predict_emotion(text):
    """run text through the emotion classifier (same as app.py)"""
    X = vectorizer.transform([text])
    return emotion_model.predict(X)[0]


def predict_behaviour(text):
    """run text through the behaviour state classifier (same as app.py)"""
    X = behaviour_vectorizer.transform([text])
    return behaviour_model.predict(X)[0]


# ---- paralysis score calculation (replicated from app.py) ----

BEHAVIOUR_WEIGHTS = {
    "avoidance": 3, "overwhelm": 2, "rumination": 1,
    "recovery": -1, "action": -2, "completion": -3,
}

EMOTION_WEIGHTS = {
    "guilty": 2, "anxious": 2, "overwhelmed": 2,
    "stressed": 1, "unmotivated": 1, "frustrated": 1, "stuck": 1,
    "tired": 0, "calm": -1, "hopeful": -2, "proud": -2,
}

PARALYSIS_KEYWORDS = ["can't", "never", "always", "hate", "impossible"]


def calc_paralysis_score(emotion, behaviour, text, same_day_negatives=0):
    """same formula as app.py calculate_paralysis_score without the DB lookup"""
    if not text or not emotion:
        return None

    raw = 0.0

    # behaviour weight
    if behaviour and behaviour in BEHAVIOUR_WEIGHTS:
        raw += BEHAVIOUR_WEIGHTS[behaviour]

    # emotion weight
    if emotion in EMOTION_WEIGHTS:
        raw += EMOTION_WEIGHTS[emotion]

    # keyword boost (capped at +3)
    lower = text.lower()
    kw_count = sum(1 for kw in PARALYSIS_KEYWORDS if kw in lower)
    raw += min(kw_count, 3)

    # frequency factor - 3+ negative entries on the same day adds +1
    neg_emotions = [e for e, w in EMOTION_WEIGHTS.items() if w >= 1]
    total_neg = same_day_negatives
    if emotion in neg_emotions:
        total_neg += 1
    if total_neg >= 3:
        raw += 1

    return round(max(-5.0, min(5.0, raw)), 1)


# ---- connect to database ----
print("\nConnecting to Supabase PostgreSQL...")
conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
cur = conn.cursor()

# bypass RLS - the postgres superuser ignores row-level security policies
# this lets us insert rows directly without needing a flask session
cur.execute("RESET ROLE;")
print("  Role reset to postgres (RLS bypassed for seeding)")

try:
    # ================================================================
    # 1. DELETE EXISTING USER (same cascade as /delete-account route)
    # ================================================================
    print(f"\nChecking for existing '{USERNAME}'...")
    cur.execute("SELECT id FROM users WHERE username = %s", (USERNAME,))
    existing = cur.fetchone()

    if existing:
        old_id = existing['id']
        print(f"  Found existing user (id={old_id}), deleting all data...")

        # children first, then parents, same order as app.py delete_account
        cur.execute(
            "DELETE FROM habit_completions WHERE habit_id IN "
            "(SELECT id FROM habits WHERE user_id = %s)", (old_id,))
        cur.execute(
            "DELETE FROM step_completions WHERE step_id IN "
            "(SELECT id FROM goal_steps WHERE user_id = %s)", (old_id,))
        cur.execute("DELETE FROM goal_steps WHERE user_id = %s", (old_id,))
        cur.execute("DELETE FROM positive_thoughts WHERE user_id = %s", (old_id,))
        cur.execute("DELETE FROM identity_beliefs WHERE user_id = %s", (old_id,))
        cur.execute("DELETE FROM todos WHERE user_id = %s", (old_id,))
        cur.execute("DELETE FROM habits WHERE user_id = %s", (old_id,))
        cur.execute("DELETE FROM journal_entries WHERE user_id = %s", (old_id,))
        cur.execute("DELETE FROM alignment_state WHERE user_id = %s", (old_id,))
        cur.execute("DELETE FROM goals WHERE user_id = %s", (old_id,))
        cur.execute("DELETE FROM users WHERE id = %s", (old_id,))
        print("  Deleted existing user and all associated data.")
    else:
        print("  No existing user found.")

    # ================================================================
    # 2. CREATE USER
    # ================================================================
    print("\nCreating user...")
    pw_hash = generate_password_hash(PASSWORD)
    # pretend the user signed up 21 days ago
    signup_ts = (datetime.now() - timedelta(days=21)).isoformat(timespec="seconds")

    cur.execute(
        "INSERT INTO users (username, password_hash, created_at, onboarding_complete, "
        "failed_login_attempts, locked_until) "
        "VALUES (%s, %s, %s, 1, 0, NULL) RETURNING id",
        (USERNAME, pw_hash, signup_ts))
    user_id = cur.fetchone()['id']
    print(f"  Created user '{USERNAME}' (id={user_id})")

    # ================================================================
    # 3. GOALS
    # ================================================================
    print("Creating goals...")
    goal_texts = [
        "Finish my final year project report to a high standard",
        "Build a consistent morning routine that supports my mental health",
        "Improve my fitness by moving my body every day",
    ]
    goal_ids = []
    for g in goal_texts:
        cur.execute(
            "INSERT INTO goals (user_id, goal_text, created_at) "
            "VALUES (%s, %s, %s) RETURNING id",
            (user_id, g, signup_ts))
        goal_ids.append(cur.fetchone()['id'])
    print(f"  Created {len(goal_ids)} goals")

    # ================================================================
    # 4. IDENTITY BELIEFS (linked to goals)
    # ================================================================
    print("Creating identity beliefs...")
    belief_data = [
        (goal_ids[0], "I am someone who shows up and does the work, even when it feels hard."),
        (goal_ids[1], "I am someone who takes care of my mind and body because they carry me through everything."),
        (goal_ids[2], "I am someone who moves my body because it helps me think clearly and feel strong."),
    ]
    belief_ids = []
    for gid, text in belief_data:
        cur.execute(
            "INSERT INTO identity_beliefs (user_id, belief_text, linked_goal_id, created_at) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (user_id, text, gid, signup_ts))
        belief_ids.append(cur.fetchone()['id'])
    print(f"  Created {len(belief_ids)} identity beliefs")

    # ================================================================
    # 5. POSITIVE THOUGHTS (linked to beliefs)
    # ================================================================
    print("Creating positive thoughts...")
    thought_data = [
        (belief_ids[0], "Every small step I take is proof that I am becoming the person I want to be."),
        (belief_ids[1], "Looking after myself is not selfish, it is what allows me to show up for everything else."),
        (belief_ids[2], "My body and my mind work together, and giving them what they need is how I grow."),
    ]
    for bid, text in thought_data:
        cur.execute(
            "INSERT INTO positive_thoughts (user_id, thought_text, linked_belief_id, created_at) "
            "VALUES (%s, %s, %s, %s)",
            (user_id, text, bid, signup_ts))
    print(f"  Created {len(thought_data)} positive thoughts")

    # ================================================================
    # 6. HABITS (3 custom + 3 sample, linked to goals)
    # ================================================================
    print("Creating habits...")
    habit_defs = [
        # (name, is_sample, linked_goal_id)
        ("Open my report and write for 10 minutes", 0, goal_ids[0]),
        ("Drink a glass of water before checking my phone in the morning", 0, goal_ids[1]),
        ("Do 5 minutes of stretching after lunch", 0, goal_ids[2]),
        ("Take 3 deep breaths before starting work", 1, goal_ids[1]),
        ("Write one thing I am grateful for", 1, goal_ids[1]),
        ("Walk for 10 minutes outside", 1, goal_ids[2]),
    ]
    habit_ids = []
    for name, is_sample, gid in habit_defs:
        cur.execute(
            "INSERT INTO habits (user_id, name, is_sample, is_hidden, linked_goal_id, created_at) "
            "VALUES (%s, %s, %s, 0, %s, %s) RETURNING id",
            (user_id, name, is_sample, gid, signup_ts))
        habit_ids.append(cur.fetchone()['id'])
    print(f"  Created {len(habit_ids)} habits (3 custom + 3 sample)")

    # ================================================================
    # 7. GOAL STEPS
    # ================================================================
    print("Creating goal steps...")
    today = date.today()

    # (goal_id, step_text, step_order, frequency, due_date, day_of_week, is_done)
    # day_of_week uses python weekday(): 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
    step_defs = [
        # goal 1: FYP report (5 steps)
        (goal_ids[0], "Open Chapter 4 and re-read what I wrote yesterday",
         1, "daily", None, None, 0),
        (goal_ids[0], "Write 200 words on the methodology section",
         2, "daily", None, None, 0),
        (goal_ids[0], "Email Dr Sloan with my draft of Chapter 5",
         3, "one-off", today.isoformat(), None, 0),
        (goal_ids[0], "Print out the rubric and tape it above my desk",
         4, "one-off", (today - timedelta(days=3)).isoformat(), None, 1),
        (goal_ids[0], "Review my literature review citations",
         5, "weekly", None, 2, 0),  # wednesday

        # goal 2: morning routine (4 steps)
        (goal_ids[1], "Put my phone in another room before bed",
         1, "daily", None, None, 0),
        (goal_ids[1], "Wake up at 7:30am without snoozing",
         2, "daily", None, None, 0),
        (goal_ids[1], "Make my bed within 5 minutes of waking up",
         3, "daily", None, None, 0),
        (goal_ids[1], "Plan tomorrow's schedule before bed",
         4, "daily", None, None, 0),

        # goal 3: fitness (4 steps)
        (goal_ids[2], "Stretch for 5 minutes after waking up",
         1, "daily", None, None, 0),
        (goal_ids[2], "Go for a 20 minute walk after lunch",
         2, "daily", None, None, 0),
        (goal_ids[2], "Do one full workout session",
         3, "weekly", None, 5, 0),  # saturday
        (goal_ids[2], "Try a new yoga video",
         4, "one-off", (today + timedelta(days=2)).isoformat(), None, 0),
    ]

    step_ids = []
    for gid, text, order, freq, due, dow, done in step_defs:
        cur.execute(
            "INSERT INTO goal_steps (goal_id, user_id, step_text, step_order, frequency, "
            "due_date, day_of_week, is_done, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (gid, user_id, text, order, freq, due, dow, done, signup_ts))
        step_ids.append(cur.fetchone()['id'])
    print(f"  Created {len(step_ids)} goal steps")

    # ================================================================
    # 8. JOURNAL ENTRIES (14 entries, ML predictions + paralysis scores)
    # ================================================================
    print("\nCreating journal entries (running ML predictions)...")

    # (text, days_ago, hour, minute)
    # ordered from oldest to newest so the temporal analysis works correctly
    entry_defs = [
        # 14 days ago - overwhelmed, opening/closing document
        (
            "I have so much to do for my final year project and I keep opening the "
            "document and closing it again. Every time I look at the rubric I feel my "
            "chest tighten. I have not made any real progress in days.",
            14, 21, 15
        ),
        # 13 days ago - frustrated, going in circles (ADDED for emotion coverage)
        (
            "I am so frustrated with myself. I wasted the whole day and got nothing "
            "done on my report. I keep going in circles in my head about what I should "
            "have done differently. I am angry and I hate that I cannot just start.",
            13, 19, 45
        ),
        # 12 days ago - unmotivated, scrolling instead of working
        (
            "I keep scrolling on my phone instead of working on my report. I have "
            "zero motivation to start. Nothing feels worth the effort and I do not "
            "care about this assignment right now.",
            12, 16, 30
        ),
        # 10 days ago - tired, exhausted and drained
        (
            "I am exhausted from staying up late trying to catch up. My brain feels "
            "foggy and I feel so tired I can barely keep my eyes open. I need to rest "
            "before I can do anything.",
            10, 14, 20
        ),
        # 9 days ago - stressed, deadlines piling up (ADDED for emotion coverage)
        (
            "The deadline is getting closer and I can feel the pressure building in "
            "my chest. I have three things due this week and I keep jumping between "
            "them without finishing any of them. My shoulders are so tense.",
            9, 20, 10
        ),
        # 8 days ago - guilty, spiralling thoughts about neglecting friends
        (
            "I cannot stop thinking about how I have not been replying to my friends. "
            "The guilt keeps going over and over in my head. I tell myself I will "
            "message them but the thought just spirals.",
            8, 22, 5
        ),
        # 6 days ago - turning point, feeling calm and steady
        (
            "I feel calm and steady today. I opened my report and read through what "
            "I had. It was not as bad as I remembered and things feel manageable.",
            6, 17, 40
        ),
        # 4 days ago - walk helped, wrote 300 words
        (
            "I went for a walk this morning and it actually helped clear my head. I "
            "came back and wrote 300 words for Chapter 4 without overthinking it.",
            4, 13, 25
        ),
        # 3 days ago - hopeful, believing in progress (ADDED for emotion coverage)
        (
            "Something shifted today. I woke up and actually felt like I could handle "
            "what is in front of me. I am starting to believe that if I just keep "
            "showing up, things will work out.",
            3, 10, 15
        ),
        # 2 days ago - proud, finished methodology section
        (
            "I finished the methodology section of Chapter 4 today. It took me three "
            "hours but I am proud that I sat with it instead of running away.",
            2, 18, 30
        ),
        # yesterday - anxious but pushed through and made progress
        (
            "I felt anxious this morning and kept worrying about failing. But I "
            "forced myself to sit down and start working on my project. I got through "
            "two tasks from my list and that helped calm the nerves a bit.",
            1, 15, 50
        ),
        # TODAY entry 1 (earliest) - overwhelmed, too much to do
        (
            "I woke up feeling overwhelmed by everything I have left to do. There is "
            "too much to do and I do not know where to start. My mind feels overloaded "
            "and I keep shutting down.",
            0, 9, 15
        ),
        # TODAY entry 2 - transitional, calming down and planning
        (
            "I took a few minutes to breathe and write down one small step. Just one "
            "thing: open Chapter 5 and re-read the intro. I feel calmer now and it "
            "feels doable.",
            0, 12, 30
        ),
        # TODAY entry 3 (latest) - proud, completed the task
        (
            "I opened Chapter 5, re-read the intro, and fixed two small things. I "
            "completed what I set out to do today and I feel proud of myself for "
            "finishing it.",
            0, 16, 0
        ),
    ]

    # track negative entries on the same day for the frequency factor
    neg_emotions = [e for e, w in EMOTION_WEIGHTS.items() if w >= 1]
    today_neg_count = 0
    journal_ids = []

    for text, days_ago, hour, minute in entry_defs:
        # build the timestamp for this entry
        entry_dt = (datetime.now() - timedelta(days=days_ago)).replace(
            hour=hour, minute=minute, second=0, microsecond=0)
        ts_str = entry_dt.isoformat(timespec="seconds")

        # run both ML models
        emotion = predict_emotion(text)
        behaviour = predict_behaviour(text)

        # calculate paralysis score with same-day frequency tracking for today
        if days_ago == 0:
            score = calc_paralysis_score(emotion, behaviour, text, today_neg_count)
            # update running count of negative entries for today
            if emotion in neg_emotions:
                today_neg_count += 1
        else:
            score = calc_paralysis_score(emotion, behaviour, text, 0)

        cur.execute(
            "INSERT INTO journal_entries "
            "(user_id, entry_text, predicted_emotion, predicted_behaviour, "
            "paralysis_score, reframe, micro_task_text, micro_task_minutes, created_at) "
            "VALUES (%s, %s, %s, %s, %s, NULL, NULL, NULL, %s) RETURNING id",
            (user_id, text, emotion, behaviour, score, ts_str))
        jid = cur.fetchone()['id']
        journal_ids.append(jid)

        label = f"{days_ago}d ago" if days_ago > 0 else "today"
        print(f"  [{label:>7}] {emotion:<12} / {behaviour:<11} / score={score}")

    print(f"  Inserted {len(journal_ids)} journal entries")

    # ================================================================
    # 9. TODOS (mix of manual and journal-linked)
    # ================================================================
    print("\nCreating todos...")
    now_str = datetime.now().isoformat(timespec="seconds")

    # (text, source, journal_entry_id, due_date, is_done)
    todo_defs = [
        ("Reply to mum's text",
         "manual", None, None, 0),
        ("Buy more coffee",
         "manual", None, None, 1),
        ("Submit project log update",
         "manual", None, today.isoformat(), 0),
        # linked to yesterday's entry (index 10 = anxious entry)
        ("Take 2 minutes to open my dissertation document and just look at it",
         "micro_task", journal_ids[10], None, 0),
        # linked to today's first entry (index 11 = overwhelmed entry)
        ("Write one rough sentence for my essay introduction",
         "micro_task", journal_ids[11], None, 1),
    ]

    for text, source, jid, due, done in todo_defs:
        cur.execute(
            "INSERT INTO todos (user_id, text, source, journal_entry_id, due_date, "
            "is_done, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (user_id, text, source, jid, due, done, now_str))

    todos_done = sum(1 for _, _, _, _, d in todo_defs if d == 1)
    print(f"  Created {len(todo_defs)} todos ({todos_done} done, "
          f"{len(todo_defs) - todos_done} pending)")

    # ================================================================
    # 10. HABIT COMPLETIONS (21 days of varied data)
    # ================================================================
    print("Creating habit completions (21 days)...")

    # which habit indices (0-5) get completed each day
    # pattern gives ~71% completion rate with realistic variation
    # some days are great (6/6), some are partial, one bad day (0/6)
    completion_pattern = [
        # day offset 0 = 20 days ago, offset 20 = today
        [0, 1, 2, 3],              # -20: 4/6
        [0, 1, 2, 3, 4],           # -19: 5/6
        [0, 2, 4, 5],              # -18: 4/6
        [1, 3, 4, 5],              # -17: 4/6
        [0, 1, 2, 3, 4, 5],        # -16: 6/6
        [0, 2, 3, 4],              # -15: 4/6
        [],                         # -14: 0/6 (bad day)
        [0, 1, 3, 5],              # -13: 4/6
        [0, 1, 2, 3, 4],           # -12: 5/6
        [1, 2, 4],                  # -11: 3/6
        [0, 1, 2, 3, 4, 5],        # -10: 6/6
        [0, 3, 5],                  # -9:  3/6
        [0, 1, 2, 4, 5],           # -8:  5/6
        [0, 2, 3, 4],              # -7:  4/6
        [0, 1, 2, 3, 4, 5],        # -6:  6/6
        [1, 3, 4],                  # -5:  3/6
        [0, 1, 2, 3, 4],           # -4:  5/6
        [0, 2, 3, 4, 5],           # -3:  5/6
        [0, 1, 2, 3, 4, 5],        # -2:  6/6
        [0, 1, 2, 4, 5],           # -1:  5/6
        [0, 2, 4],                  # 0 (today): 3/6
    ]

    total_habit_completions = 0
    for offset, indices in enumerate(completion_pattern):
        days_ago = 20 - offset
        comp_day = datetime.now() - timedelta(days=days_ago)
        for i, habit_idx in enumerate(indices):
            # stagger completion times through the morning
            comp_ts = comp_day.replace(
                hour=7 + i, minute=30, second=0, microsecond=0)
            cur.execute(
                "INSERT INTO habit_completions (habit_id, completed_at) "
                "VALUES (%s, %s)",
                (habit_ids[habit_idx], comp_ts.isoformat(timespec="seconds")))
            total_habit_completions += 1

    rate = total_habit_completions / (21 * 6) * 100
    print(f"  Created {total_habit_completions} completions "
          f"({rate:.0f}% rate over 21 days)")

    # ================================================================
    # 11. STEP COMPLETIONS (daily steps that are "done" for today + history)
    # ================================================================
    print("Creating step completions...")

    # "Make my bed within 5 minutes of waking up" = step_ids[7] (goal 2, step 3)
    # "Stretch for 5 minutes after waking up" = step_ids[9] (goal 3, step 1)
    make_bed_id = step_ids[7]
    stretch_id = step_ids[9]

    # scattered completions over the past 14 days to show progress
    make_bed_days_ago = [0, 1, 2, 4, 5, 7, 10, 12]   # 8 completions
    stretch_days_ago = [0, 1, 3, 5, 8, 11, 13]         # 7 completions

    step_comp_count = 0
    for d in make_bed_days_ago:
        cur.execute(
            "INSERT INTO step_completions (step_id, user_id, completed_date) "
            "VALUES (%s, %s, %s)",
            (make_bed_id, user_id, (today - timedelta(days=d)).isoformat()))
        step_comp_count += 1

    for d in stretch_days_ago:
        cur.execute(
            "INSERT INTO step_completions (step_id, user_id, completed_date) "
            "VALUES (%s, %s, %s)",
            (stretch_id, user_id, (today - timedelta(days=d)).isoformat()))
        step_comp_count += 1

    print(f"  Created {step_comp_count} step completions")

    # ================================================================
    # 12. ALIGNMENT STATE
    # ================================================================
    print("\nCalculating alignment state...")

    # each completed action adds +1 (same as app.py)
    alignment_score = (
        len(journal_ids)            # journal entries
        + total_habit_completions   # habit completions
        + todos_done                # completed todos
        + step_comp_count           # step completions
    )

    # emotional streak = consecutive days with journal entries from today backwards
    journal_days = set()
    for _, days_ago, _, _ in entry_defs:
        journal_days.add(days_ago)

    streak = 0
    for d in range(0, 30):
        if d in journal_days:
            streak += 1
        else:
            break

    cur.execute(
        "INSERT INTO alignment_state "
        "(user_id, alignment_score, emotional_streak, last_journal_date) "
        "VALUES (%s, %s, %s, %s)",
        (user_id, alignment_score, streak, today.isoformat()))

    print(f"  Alignment score: {alignment_score}")
    print(f"    (journals={len(journal_ids)}, habits={total_habit_completions}, "
          f"todos={todos_done}, steps={step_comp_count})")
    print(f"  Emotional streak: {streak} days")
    print(f"  Last journal date: {today.isoformat()}")

    # ================================================================
    # COMMIT - all changes go in as a single transaction
    # ================================================================
    conn.commit()

    # ================================================================
    # VERIFY - quick count check against the database
    # ================================================================
    print("\nVerifying inserted data...")
    counts = {}
    for table in ['goals', 'identity_beliefs', 'positive_thoughts', 'habits',
                   'goal_steps', 'journal_entries', 'todos']:
        cur.execute(f"SELECT COUNT(*) as cnt FROM {table} WHERE user_id = %s",
                    (user_id,))
        counts[table] = cur.fetchone()['cnt']

    cur.execute(
        "SELECT COUNT(*) as cnt FROM habit_completions WHERE habit_id IN "
        "(SELECT id FROM habits WHERE user_id = %s)", (user_id,))
    counts['habit_completions'] = cur.fetchone()['cnt']

    cur.execute(
        "SELECT COUNT(*) as cnt FROM step_completions WHERE user_id = %s",
        (user_id,))
    counts['step_completions'] = cur.fetchone()['cnt']

    for table, cnt in counts.items():
        print(f"  {table}: {cnt} rows")

    print("\n" + "=" * 60)
    print("Test user created successfully.")
    print(f"Login with username: {USERNAME} and password: {PASSWORD}")
    print("=" * 60)

except Exception as e:
    conn.rollback()
    print(f"\nERROR: Script failed, all changes rolled back.")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    cur.close()
    conn.close()
