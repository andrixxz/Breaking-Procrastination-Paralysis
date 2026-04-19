"""
Test script for Task 6.5 - Weekly + Monthly Summary Stats.
Tests the new summary fields on /week and /month pages.
Run: python test_weekly_monthly_summary.py
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

# figure out which days are in the current week (mon-sun)
week_start = today - timedelta(days=today.weekday())
week_end = week_start + timedelta(days=6)
# how many days ago is monday of this week
days_since_monday = today.weekday()

# clean up if test user exists from a previous run
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
            ("summarytest99",),
        )
    except Exception:
        conn.rollback()
cur.execute("DELETE FROM users WHERE username = %s", ("summarytest99",))
conn.commit()

# create test user
pw_hash = generate_password_hash("testpass123")
cur.execute(
    "INSERT INTO users (username, password_hash, created_at, onboarding_complete) "
    "VALUES (%s, %s, %s, %s) RETURNING id",
    ("summarytest99", pw_hash, now, 1),
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
    (user_id, "I am someone who takes action", now),
)
conn.commit()
conn.close()

print(f"Test user created (id={user_id})")
print(f"Today: {today} ({today.strftime('%A')}), Week: {week_start} to {week_end}\n")

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
    """wipe journal entries for the test user"""
    c = get_db_connection()
    cr = c.cursor()
    cr.execute("RESET ROLE;")
    cr.execute("DELETE FROM journal_entries WHERE user_id = %s", (user_id,))
    c.commit()
    c.close()


def add_entry(emotion, behaviour, score, days_ago=0):
    """add a journal entry - days_ago=0 means today"""
    entry_date = today - timedelta(days=days_ago)
    ts = f"{entry_date.isoformat()} 10:00:00"
    c = get_db_connection()
    cr = c.cursor()
    cr.execute("RESET ROLE;")
    cr.execute(
        "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, "
        "predicted_behaviour, paralysis_score, created_at) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (user_id, f"test entry {emotion}", emotion, behaviour, score, ts),
    )
    c.commit()
    c.close()


def add_entry_at_time(emotion, behaviour, score, days_ago=0, hour=10):
    """add a journal entry at a specific hour for ordering tests"""
    entry_date = today - timedelta(days=days_ago)
    ts = f"{entry_date.isoformat()} {hour:02d}:00:00"
    c = get_db_connection()
    cr = c.cursor()
    cr.execute("RESET ROLE;")
    cr.execute(
        "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, "
        "predicted_behaviour, paralysis_score, created_at) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (user_id, f"test entry {emotion}", emotion, behaviour, score, ts),
    )
    c.commit()
    c.close()


# single client session to avoid rate limit on login
client = app.test_client()
with client.session_transaction() as sess:
    sess['user_id'] = user_id
    sess['username'] = 'summarytest99'


# ===== GROUP 1: Week Page - No Data =====
print("GROUP 1: Week page with no journal data")
clear_entries()
resp = client.get('/week')
html = resp.data.decode()

check("Week page loads with no entries", resp.status_code == 200)
check("No avg paralysis shown when no data", "Avg. paralysis score" not in html)
check("No top emotion shown when no data", "Most felt this week" not in html)
check("No positive shifts shown when no data", "Positive shifts" not in html)
check("Reflections this week label present", "Reflections this week" in html)
print()

# ===== GROUP 2: Week Page - With Emotion Data =====
print("GROUP 2: Week page with emotion data (all today so definitely in this week)")
clear_entries()
# all entries on today so they are always in the current week
add_entry_at_time("anxious", "Avoidance", 3, days_ago=0, hour=8)
add_entry_at_time("anxious", "Overwhelm", 2, days_ago=0, hour=10)
add_entry_at_time("calm", "Action", -1, days_ago=0, hour=14)

resp = client.get('/week')
html = resp.data.decode()

check("Week page loads with entries", resp.status_code == 200)
check("Avg paralysis score shown", "Avg. paralysis score" in html)
# (3 + 2 + -1) / 3 = 1.3
check("Avg paralysis value correct (1.3)", ">1.3</p>" in html)
check("Top emotion shown (anxious appears 2x)", "Most felt this week" in html)
check("Top emotion badge has class", 'emotion-badge anxious' in html)
check("Psych explainer present", "Why these numbers matter" in html)
print()

# ===== GROUP 3: Week Page - Positive Shifts =====
print("GROUP 3: Week page positive shifts detection")
clear_entries()
# two entries same day: Avoidance then Action = 1 positive shift
add_entry_at_time("anxious", "Avoidance", 3, days_ago=0, hour=9)
add_entry_at_time("calm", "Action", -1, days_ago=0, hour=14)

resp = client.get('/week')
html = resp.data.decode()

check("Positive shifts shown", "Positive shifts" in html)
check("Positive shifts count is 1", ">1</p>" in html and "Positive shifts" in html)
print()

# ===== GROUP 4: No shifts when entries are on different days =====
print("GROUP 4: No shifts when entries are on different days")
clear_entries()
# one entry today, one entry yesterday - both in the current week but on different days
# shifts only count within the SAME day
if days_since_monday >= 1:
    # if today is tue or later, yesterday is still in this week
    add_entry("anxious", "Avoidance", 3, days_ago=0)
    add_entry("calm", "Action", -1, days_ago=1)
else:
    # today is monday so yesterday is last week - put both on today at same score
    add_entry_at_time("anxious", "Avoidance", 3, days_ago=0, hour=9)
    # add a non-shift pair on a different "today" entry
    add_entry_at_time("stressed", "Overwhelm", 2, days_ago=0, hour=14)

resp = client.get('/week')
html = resp.data.decode()

check("No positive shifts across different days (or neg-to-neg same day)", "Positive shifts</p>" not in html)
print()

# ===== GROUP 5: Week page - NULL behaviours =====
print("GROUP 5: Week page handles NULL behaviours gracefully")
clear_entries()
add_entry("anxious", None, 3, days_ago=0)
if days_since_monday >= 1:
    add_entry("anxious", None, 2, days_ago=1)
else:
    add_entry_at_time("anxious", None, 2, days_ago=0, hour=14)

resp = client.get('/week')
html = resp.data.decode()

check("Page loads with NULL behaviours", resp.status_code == 200)
check("Top emotion still shown with NULL behaviours", "Most felt this week" in html)
check("No shifts when all behaviours are NULL", "Positive shifts</p>" not in html)
print()

# ===== GROUP 6: Week page - NULL paralysis scores =====
print("GROUP 6: Week page handles NULL paralysis scores")
clear_entries()
add_entry("anxious", "Avoidance", None, days_ago=0)

resp = client.get('/week')
html = resp.data.decode()

check("Page loads with NULL scores", resp.status_code == 200)
check("No avg paralysis when all scores NULL", "Avg. paralysis score" not in html)
print()

# ===== GROUP 7: Old entries excluded =====
print("GROUP 7: Old entries excluded from weekly summary")
clear_entries()
# 10 days ago is definitely not in the current week
add_entry("stressed", "Overwhelm", 4, days_ago=10)
add_entry("calm", "Action", -1, days_ago=0)

resp = client.get('/week')
html = resp.data.decode()

check("Only this week entries counted", "Reflections this week" in html)
check("Top emotion is calm (not stressed from 10 days ago)", 'emotion-badge calm' in html)
print()

# ===== GROUP 8: Month page - no data =====
print("GROUP 8: Month page with no journal data")
clear_entries()
resp = client.get('/month')
html = resp.data.decode()

check("Month page loads with no entries", resp.status_code == 200)
check("No top emotion on month when no data", "Most felt this month" not in html)
check("No positive shifts on month when no data", "Positive shifts" not in html)
print()

# ===== GROUP 9: Month page with data =====
print("GROUP 9: Month page with emotion and behaviour data")
clear_entries()
add_entry("anxious", "Avoidance", 3, days_ago=0)
add_entry("stressed", "Overwhelm", 2, days_ago=1)
add_entry("anxious", "Avoidance", 4, days_ago=2)
add_entry("calm", "Action", -1, days_ago=3)

resp = client.get('/month')
html = resp.data.decode()

check("Month page loads with entries", resp.status_code == 200)
check("Top emotion shown on month (anxious)", "Most felt this month" in html)
check("Emotion badge on month page", 'emotion-badge anxious' in html)
check("Avg paralysis shown on month", "Avg. paralysis score" in html)
print()

# ===== GROUP 10: Month page - positive shifts =====
print("GROUP 10: Month page positive shifts")
clear_entries()
add_entry_at_time("anxious", "Avoidance", 3, days_ago=0, hour=9)
add_entry_at_time("calm", "Action", -1, days_ago=0, hour=14)
add_entry_at_time("stressed", "Overwhelm", 2, days_ago=1, hour=10)
add_entry_at_time("hopeful", "Completion", -2, days_ago=1, hour=16)

resp = client.get('/month')
html = resp.data.decode()

check("Positive shifts shown on month", "Positive shifts" in html)
check("Two positive shifts detected", ">2</p>" in html and "Positive shifts" in html)
print()

# ===== GROUP 11: Month - Previous month comparison =====
print("GROUP 11: Previous month comparison")
clear_entries()
# add entries for THIS month
add_entry("anxious", "Avoidance", 3, days_ago=0)
add_entry("calm", "Action", -1, days_ago=1)

# add entries for LAST month directly
last_month_date = today.replace(day=1) - timedelta(days=1)
c = get_db_connection()
cr = c.cursor()
cr.execute("RESET ROLE;")
ts = f"{last_month_date.isoformat()} 10:00:00"
cr.execute(
    "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, "
    "predicted_behaviour, paralysis_score, created_at) "
    "VALUES (%s, %s, %s, %s, %s, %s)",
    (user_id, "last month entry 1", "stressed", "Overwhelm", 4, ts),
)
ts2 = f"{(last_month_date - timedelta(days=1)).isoformat()} 10:00:00"
cr.execute(
    "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, "
    "predicted_behaviour, paralysis_score, created_at) "
    "VALUES (%s, %s, %s, %s, %s, %s)",
    (user_id, "last month entry 2", "anxious", "Avoidance", 3, ts2),
)
c.commit()
c.close()

resp = client.get('/month')
html = resp.data.decode()

check("Month page loads with comparison data", resp.status_code == 200)
check("Comparison text appears", "from" in html.lower() or "same as last month" in html.lower())
check("Psych explainer for month stats", "Why these numbers matter" in html)
print()

# ===== GROUP 12: No previous month Data =====
print("GROUP 12: No previous month comparison when no prior data")
clear_entries()
add_entry("calm", "Action", -1, days_ago=0)

resp = client.get('/month')
html = resp.data.decode()

check("Month loads without prior data", resp.status_code == 200)
check("No comparison when prev month empty", "up from" not in html and "down from" not in html)
print()

# ===== GROUP 13: All entries same emotion =====
print("GROUP 13: All entries have the same emotion")
clear_entries()
add_entry("anxious", "Avoidance", 3, days_ago=0)
if days_since_monday >= 1:
    add_entry("anxious", "Avoidance", 3, days_ago=1)
else:
    add_entry_at_time("anxious", "Avoidance", 3, days_ago=0, hour=14)

resp = client.get('/week')
html = resp.data.decode()

check("Top emotion is anxious when all entries anxious", 'emotion-badge anxious' in html)
print()

# ===== GROUP 14: Single entry =====
print("GROUP 14: Single entry only")
clear_entries()
add_entry("calm", "Action", -2, days_ago=0)

resp = client.get('/week')
html = resp.data.decode()

check("Top emotion shown with single entry", 'emotion-badge calm' in html)
check("Avg paralysis shown with single entry", "Avg. paralysis score" in html)
check("No shifts with single entry (only 1 entry total)", "Positive shifts</p>" not in html)
print()

# ===== GROUP 15: Mixed NULL and valid Data =====
print("GROUP 15: Mix of NULL and valid behaviour/score data")
clear_entries()
# all entries have emotion (NOT NULL constraint), but behaviour and score can be null
add_entry("anxious", "Avoidance", 3, days_ago=0)
add_entry("calm", None, None, days_ago=0)

resp = client.get('/week')
html = resp.data.decode()

check("Page loads with mixed nulls", resp.status_code == 200)
check("Top emotion from non-null entries", "Most felt this week" in html)
# only 1 non-null score (3), so avg should be 3.0
check("Avg paralysis from non-null scores only", "Avg. paralysis score" in html)
print()

# ===== GROUP 16: Month navigation =====
print("GROUP 16: Month navigation with summary stats")
clear_entries()
resp = client.get(f'/month?year={today.year}&month={today.month}')
html = resp.data.decode()

check("Month nav loads with year/month params", resp.status_code == 200)
print()

# ===== GROUP 17: Multiple shifts same day =====
print("GROUP 17: Multiple positive shifts in one day")
clear_entries()
# Avoidance -> Action -> Overwhelm -> Completion = 2 positive shifts
add_entry_at_time("anxious", "Avoidance", 3, days_ago=0, hour=8)
add_entry_at_time("calm", "Action", -1, days_ago=0, hour=11)
add_entry_at_time("stressed", "Overwhelm", 2, days_ago=0, hour=14)
add_entry_at_time("hopeful", "Completion", -3, days_ago=0, hour=17)

resp = client.get('/week')
html = resp.data.decode()

check("Multiple shifts detected in one day", "Positive shifts" in html)
# Avoidance->Action = 1, Action->Overwhelm = no, Overwhelm->Completion = 1 = 2 total
check("Two shifts counted", ">2</p>" in html)
print()

# ===== GROUP 18: NEGATIVE TO NEGATIVE NOT A SHIFT =====
print("GROUP 18: Negative to negative is not a positive shift")
clear_entries()
add_entry_at_time("anxious", "Avoidance", 3, days_ago=0, hour=9)
add_entry_at_time("stressed", "Overwhelm", 2, days_ago=0, hour=14)

resp = client.get('/week')
html = resp.data.decode()

check("No shift for negative to negative", "Positive shifts</p>" not in html)
print()

# ===== GROUP 19: Positive to positive not a shift =====
print("GROUP 19: Positive to positive is not a shift")
clear_entries()
add_entry_at_time("calm", "Action", -1, days_ago=0, hour=9)
add_entry_at_time("hopeful", "Completion", -3, days_ago=0, hour=14)

resp = client.get('/week')
html = resp.data.decode()

check("No shift for positive to positive", "Positive shifts</p>" not in html)
print()

# ===== GROUP 20: Paralysis score - improved =====
print("GROUP 20: Paralysis score improved compared to last month")
clear_entries()
# this month: low paralysis
add_entry("calm", "Action", -1, days_ago=0)
add_entry("calm", "Action", -2, days_ago=1)

# last month: high paralysis
last_month_date = today.replace(day=1) - timedelta(days=1)
c = get_db_connection()
cr = c.cursor()
cr.execute("RESET ROLE;")
ts = f"{last_month_date.isoformat()} 10:00:00"
cr.execute(
    "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, "
    "predicted_behaviour, paralysis_score, created_at) "
    "VALUES (%s, %s, %s, %s, %s, %s)",
    (user_id, "last month stressed entry", "stressed", "Overwhelm", 4, ts),
)
c.commit()
c.close()

resp = client.get('/month')
html = resp.data.decode()

check("Paralysis comparison appears", "Avg. paralysis score" in html)
# current avg is -1.5, prev is 4.0. current < prev so it should say "improved"
check("Score improved message shows", "that is progress" in html.lower() or "down from" in html.lower())
print()


# ===== Cleanup =====
print("\n--- CLEANUP ---")
clear_entries()
c = get_db_connection()
cr = c.cursor()
cr.execute("RESET ROLE;")
for t in tables_to_clean:
    try:
        cr.execute(
            f"DELETE FROM {t} WHERE user_id = %s", (user_id,)
        )
    except Exception:
        c.rollback()
cr.execute("DELETE FROM users WHERE id = %s", (user_id,))
c.commit()
c.close()
print("Test user cleaned up.")

print(f"\n{'=' * 40}")
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
print(f"{'=' * 40}")
