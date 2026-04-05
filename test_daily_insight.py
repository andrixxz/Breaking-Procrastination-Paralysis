"""
Test script for Task 6.4 - Daily Insight Display on Dashboard + Today page.
Run: python test_daily_insight.py
"""
import os
os.environ['FLASK_ENV'] = 'development'

from app import app, get_db_connection
from werkzeug.security import generate_password_hash
from datetime import datetime, date, timedelta

app.config['WTF_CSRF_ENABLED'] = False
app.config['TESTING'] = True

conn = get_db_connection()
cur = conn.cursor()
now = datetime.now().isoformat(timespec="seconds")
today = date.today()

# clean up if user exists
cur.execute("RESET ROLE;")
tables_to_clean = [
    "habit_completions", "step_completions", "goal_steps",
    "positive_thoughts", "identity_beliefs", "todos",
    "habits", "journal_entries", "alignment_state", "goals",
]
for t in tables_to_clean:
    try:
        cur.execute(
            f"DELETE FROM {t} WHERE user_id IN (SELECT id FROM users WHERE username = %s)",
            ("insighttest99",),
        )
    except Exception:
        conn.rollback()
cur.execute("DELETE FROM users WHERE username = %s", ("insighttest99",))
conn.commit()

# create user
pw_hash = generate_password_hash("testpass123")
cur.execute(
    "INSERT INTO users (username, password_hash, created_at, onboarding_complete) "
    "VALUES (%s, %s, %s, %s) RETURNING id",
    ("insighttest99", pw_hash, now, 1),
)
user_id = cur.fetchone()["id"]
cur.execute(
    "INSERT INTO alignment_state (user_id, alignment_score, emotional_streak, last_journal_date) "
    "VALUES (%s, 0, 0, %s)",
    (user_id, ""),
)
cur.execute(
    "INSERT INTO goals (user_id, goal_text, created_at) VALUES (%s, %s, %s)",
    (user_id, "test goal", now),
)
cur.execute(
    "INSERT INTO identity_beliefs (user_id, belief_text, created_at) VALUES (%s, %s, %s)",
    (user_id, "I am someone who keeps going", now),
)
conn.commit()
conn.close()

print(f"Test user created (id={user_id})\n")

passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")


def clear_entries():
    """wipe all journal entries for the test user"""
    c = get_db_connection()
    cr = c.cursor()
    cr.execute("RESET ROLE;")
    cr.execute("DELETE FROM journal_entries WHERE user_id = %s", (user_id,))
    c.commit()
    c.close()


def insert_entries(entries):
    """insert a list of (timestamp, emotion, behaviour, paralysis_score) tuples"""
    c = get_db_connection()
    cr = c.cursor()
    cr.execute("RESET ROLE;")
    for ts, emo, bh, ps in entries:
        if bh is not None:
            cr.execute(
                "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, "
                "predicted_behaviour, paralysis_score, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (user_id, f"test - {emo}", emo, bh, ps, ts),
            )
        else:
            cr.execute(
                "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, "
                "created_at) VALUES (%s, %s, %s, %s)",
                (user_id, f"test - {emo}", emo, ts),
            )
    c.commit()
    c.close()


# use a single test client to avoid hitting rate limiter on /login
with app.test_client() as c:
    c.post("/login", data={"username": "insighttest99", "password": "testpass123"})

    # ========================================================
    # TEST GROUP 1: No entries - no insight shown
    # ========================================================
    print("--- Test Group 1: No entries ---")
    clear_entries()

    resp = c.get("/dashboard")
    html = resp.data.decode("utf-8")
    check("Dashboard loads with no entries", resp.status_code == 200)
    check("No insight shown when no entries", "daily-insight" not in html)

    resp = c.get("/today")
    html = resp.data.decode("utf-8")
    check("Today loads with no entries", resp.status_code == 200)
    check("No insight on today with no entries", "daily-insight" not in html)

    # ========================================================
    # TEST GROUP 2: Single entry - no insight (need >= 2)
    # ========================================================
    print("\n--- Test Group 2: Single entry ---")
    clear_entries()
    insert_entries([
        (today.isoformat() + "T10:00:00", "anxious", "avoidance", 3.0),
    ])

    resp = c.get("/dashboard")
    html = resp.data.decode("utf-8")
    check("No insight with single entry (dashboard)", "daily-insight" not in html)

    resp = c.get("/today")
    html = resp.data.decode("utf-8")
    check("No insight with single entry (today)", "daily-insight" not in html)

    # ========================================================
    # TEST GROUP 3: Positive transition (negative -> positive)
    # ========================================================
    print("\n--- Test Group 3: Positive transition ---")
    clear_entries()
    insert_entries([
        (today.isoformat() + "T09:00:00", "overwhelmed", "avoidance", 4.0),
        (today.isoformat() + "T15:00:00", "calm", "action", -2.0),
    ])

    resp = c.get("/dashboard")
    html = resp.data.decode("utf-8")
    check("Positive transition detected", "moved from feeling overwhelmed to feeling calm" in html)
    check("Positive insight CSS class", "insight-positive" in html)
    check("Psych explainer present", "Why this works" in html)

    resp = c.get("/today")
    html = resp.data.decode("utf-8")
    check("Positive transition on today page", "moved from feeling overwhelmed to feeling calm" in html)
    check("Today's insight label on today page", "Today&#39;s insight" in html or "Today's insight" in html)

    # ========================================================
    # TEST GROUP 4: Negative loop (same negative 3+ times)
    # ========================================================
    print("\n--- Test Group 4: Negative loop ---")
    clear_entries()
    insert_entries([
        (today.isoformat() + "T08:00:00", "anxious", "avoidance", 3.0),
        (today.isoformat() + "T11:00:00", "anxious", "overwhelm", 3.5),
        (today.isoformat() + "T14:00:00", "anxious", "avoidance", 2.5),
    ])

    resp = c.get("/dashboard")
    html = resp.data.decode("utf-8")
    check("Negative loop detected", "sitting with feeling anxious" in html)
    check("Loop CSS class", "insight-loop" in html)

    resp = c.get("/today")
    html = resp.data.decode("utf-8")
    check("Negative loop on today page", "sitting with feeling anxious" in html)

    # ========================================================
    # TEST GROUP 5: Behaviour loop (same neg behaviour 3+)
    # ========================================================
    print("\n--- Test Group 5: Behaviour loop ---")
    clear_entries()
    insert_entries([
        (today.isoformat() + "T08:00:00", "overwhelmed", "avoidance", 3.0),
        (today.isoformat() + "T11:00:00", "stressed", "avoidance", 2.5),
        (today.isoformat() + "T14:00:00", "anxious", "avoidance", 2.0),
    ])

    resp = c.get("/dashboard")
    html = resp.data.decode("utf-8")
    check("Behaviour loop detected", "pattern of avoidance" in html)
    check("Behaviour loop CSS class", "insight-loop" in html)

    # ========================================================
    # TEST GROUP 6: Behaviour shift (negative -> positive beh)
    # ========================================================
    print("\n--- Test Group 6: Behaviour shift ---")
    clear_entries()
    insert_entries([
        (today.isoformat() + "T09:00:00", "stressed", "avoidance", 3.0),
        (today.isoformat() + "T15:00:00", "stressed", "action", -1.0),
    ])

    resp = c.get("/dashboard")
    html = resp.data.decode("utf-8")
    check("Behaviour shift detected", "shifted from avoidance to action" in html)
    check("Behaviour shift positive CSS", "insight-positive" in html)

    # ========================================================
    # TEST GROUP 7: Paralysis score improvement (drop of 2+)
    # ========================================================
    print("\n--- Test Group 7: Score improvement ---")
    clear_entries()
    insert_entries([
        (today.isoformat() + "T09:00:00", "stressed", "overwhelm", 4.0),
        (today.isoformat() + "T15:00:00", "stressed", "overwhelm", 1.0),
    ])

    resp = c.get("/dashboard")
    html = resp.data.decode("utf-8")
    check("Score improvement detected", "paralysis score dropped" in html)
    check("Score improvement positive CSS", "insight-positive" in html)

    # ========================================================
    # TEST GROUP 8: Emotion change (not clearly pos/neg)
    # ========================================================
    print("\n--- Test Group 8: Emotion change ---")
    clear_entries()
    insert_entries([
        (today.isoformat() + "T09:00:00", "anxious", "avoidance", 3.0),
        (today.isoformat() + "T15:00:00", "stressed", "avoidance", 2.0),
    ])

    resp = c.get("/dashboard")
    html = resp.data.decode("utf-8")
    check("Emotion change detected", "moved from feeling anxious to feeling stressed" in html)
    check("Emotion change neutral CSS", "insight-neutral" in html)

    # ========================================================
    # TEST GROUP 9: Positive consistency (same pos 2 times)
    # ========================================================
    print("\n--- Test Group 9: Positive consistency ---")
    clear_entries()
    insert_entries([
        (today.isoformat() + "T09:00:00", "calm", "action", -2.0),
        (today.isoformat() + "T15:00:00", "calm", "action", -2.0),
    ])

    resp = c.get("/dashboard")
    html = resp.data.decode("utf-8")
    check("Positive consistency detected", "stayed in a calm space" in html)
    check("Positive consistency CSS", "insight-positive" in html)

    # ========================================================
    # TEST GROUP 10: Processing (same negative 2 times)
    # ========================================================
    print("\n--- Test Group 10: Processing ---")
    clear_entries()
    insert_entries([
        (today.isoformat() + "T09:00:00", "stressed", "avoidance", 2.0),
        (today.isoformat() + "T15:00:00", "stressed", "avoidance", 2.0),
    ])

    resp = c.get("/dashboard")
    html = resp.data.decode("utf-8")
    check("Processing detected", "showed up twice today while feeling stressed" in html)
    check("Processing neutral CSS", "insight-neutral" in html)

    # ========================================================
    # TEST GROUP 11: General fallback (3+ same positive emo)
    # ========================================================
    print("\n--- Test Group 11: General fallback ---")
    clear_entries()
    insert_entries([
        (today.isoformat() + "T08:00:00", "calm", "action", -2.0),
        (today.isoformat() + "T12:00:00", "calm", "action", -2.0),
        (today.isoformat() + "T16:00:00", "calm", "action", -2.0),
    ])

    resp = c.get("/dashboard")
    html = resp.data.decode("utf-8")
    check("General fallback detected", "journaled 3 times today" in html)
    check("General fallback neutral CSS", "insight-neutral" in html)

    # ========================================================
    # TEST GROUP 12: NULL behaviours handled gracefully
    # ========================================================
    print("\n--- Test Group 12: NULL behaviours ---")
    clear_entries()
    insert_entries([
        (today.isoformat() + "T09:00:00", "overwhelmed", None, None),
        (today.isoformat() + "T15:00:00", "calm", None, None),
    ])

    resp = c.get("/dashboard")
    html = resp.data.decode("utf-8")
    check("Positive transition with NULL behaviours", "moved from feeling overwhelmed to feeling calm" in html)
    check("Page loads fine with NULL behaviours", resp.status_code == 200)

    # ========================================================
    # TEST GROUP 13: Only yesterday entries - no insight
    # ========================================================
    print("\n--- Test Group 13: Yesterday only ---")
    clear_entries()
    yesterday = today - timedelta(days=1)
    insert_entries([
        (yesterday.isoformat() + "T09:00:00", "anxious", "avoidance", 3.0),
        (yesterday.isoformat() + "T15:00:00", "calm", "action", -2.0),
    ])

    resp = c.get("/dashboard")
    html = resp.data.decode("utf-8")
    check("No insight when entries are from yesterday", "daily-insight" not in html)

    resp = c.get("/today")
    html = resp.data.decode("utf-8")
    check("No insight on today page for yesterday entries", "daily-insight" not in html)

    # ========================================================
    # TEST GROUP 14: Dashboard + Today show same insight
    # ========================================================
    print("\n--- Test Group 14: Dashboard + Today consistency ---")
    clear_entries()
    insert_entries([
        (today.isoformat() + "T09:00:00", "frustrated", "avoidance", 3.0),
        (today.isoformat() + "T16:00:00", "hopeful", "recovery", -1.0),
    ])

    resp_dash = c.get("/dashboard")
    dash_html = resp_dash.data.decode("utf-8")

    resp_today = c.get("/today")
    today_html = resp_today.data.decode("utf-8")

    expected_msg = "moved from feeling frustrated to feeling hopeful"
    check("Dashboard shows positive transition", expected_msg in dash_html)
    check("Today shows same positive transition", expected_msg in today_html)
    check("Both pages have insight label", "daily-insight-label" in dash_html and "daily-insight-label" in today_html)

    # ========================================================
    # TEST GROUP 15: Insight structure on today page
    # ========================================================
    print("\n--- Test Group 15: Today page structure ---")
    clear_entries()
    insert_entries([
        (today.isoformat() + "T09:00:00", "overwhelmed", "overwhelm", 4.0),
        (today.isoformat() + "T15:00:00", "proud", "completion", -3.0),
    ])

    resp = c.get("/today")
    html = resp.data.decode("utf-8")
    check("Today has daily-insight div", "daily-insight" in html)
    check("Today has insight label", "daily-insight-label" in html)
    check("Today has insight text", "daily-insight-text" in html)
    check("Today has psych explainer", "psych-explainer" in html)
    check("Today has Why this works", "Why this works" in html)
    check("Today has fade-in-up animation", "fade-in-up" in html)

    # ========================================================
    # TEST GROUP 16: Priority ordering - loop beats transition
    # ========================================================
    print("\n--- Test Group 16: Priority ordering ---")
    clear_entries()
    insert_entries([
        (today.isoformat() + "T08:00:00", "anxious", "avoidance", 4.0),
        (today.isoformat() + "T11:00:00", "anxious", "overwhelm", 3.5),
        (today.isoformat() + "T14:00:00", "anxious", "avoidance", 3.0),
        (today.isoformat() + "T17:00:00", "calm", "action", -2.0),
    ])

    resp = c.get("/dashboard")
    html = resp.data.decode("utf-8")
    check("Negative loop takes priority over transition", "sitting with feeling anxious" in html)

    # ========================================================
    # TEST GROUP 17: tired emotion (not pos, not neg)
    # ========================================================
    print("\n--- Test Group 17: Tired (neutral emotion) ---")
    clear_entries()
    insert_entries([
        (today.isoformat() + "T09:00:00", "tired", "avoidance", 1.0),
        (today.isoformat() + "T15:00:00", "tired", "avoidance", 1.0),
    ])

    resp = c.get("/dashboard")
    html = resp.data.decode("utf-8")
    check("Tired treated as processing", "showed up twice today while feeling tired" in html)

    # ========================================================
    # TEST GROUP 18: CSS classes present on today page
    # ========================================================
    print("\n--- Test Group 18: CSS classes ---")
    clear_entries()
    insert_entries([
        (today.isoformat() + "T09:00:00", "overwhelmed", "avoidance", 4.0),
        (today.isoformat() + "T15:00:00", "calm", "action", -2.0),
    ])

    resp = c.get("/today")
    html = resp.data.decode("utf-8")
    check("daily-insight class on today page", 'class="daily-insight' in html)
    check("daily-insight-label class", "daily-insight-label" in html)
    check("daily-insight-text class", "daily-insight-text" in html)


# clean up
conn = get_db_connection()
cur = conn.cursor()
cur.execute("RESET ROLE;")
cur.execute("DELETE FROM journal_entries WHERE user_id = %s", (user_id,))
cur.execute("DELETE FROM identity_beliefs WHERE user_id = %s", (user_id,))
cur.execute("DELETE FROM alignment_state WHERE user_id = %s", (user_id,))
cur.execute("DELETE FROM goals WHERE user_id = %s", (user_id,))
cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
conn.commit()
conn.close()

print(f"\nTest user cleaned up")
print(f"\nResults: {passed} passed, {failed} failed out of {passed + failed} tests")

if failed > 0:
    print("SOME TESTS FAILED")
else:
    print("ALL TESTS PASSED")
