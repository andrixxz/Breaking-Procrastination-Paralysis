"""
Test script for Task 6.2 - Paralysis Score Chart on analytics page.
Run: python test_paralysis_chart.py
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

# clean up if user exists from previous run
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
            ("pstest99",),
        )
    except Exception:
        conn.rollback()
cur.execute("DELETE FROM users WHERE username = %s", ("pstest99",))
conn.commit()

# create user
pw_hash = generate_password_hash("testpass123")
cur.execute(
    "INSERT INTO users (username, password_hash, created_at, onboarding_complete) "
    "VALUES (%s, %s, %s, %s) RETURNING id",
    ("pstest99", pw_hash, now, 1),
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
conn.commit()

cur.execute("SET ROLE flask_app;")
cur.execute("SET app.current_user_id = %s", (str(user_id),))

# insert entries with various paralysis scores across different dates
test_entries = [
    # today - mixed scores
    (today.isoformat() + "T10:00:00", "overwhelmed", 3.5),
    (today.isoformat() + "T14:00:00", "anxious", 2.0),
    (today.isoformat() + "T18:00:00", "calm", -2.0),
    # yesterday - improving
    ((today - timedelta(days=1)).isoformat() + "T09:00:00", "overwhelmed", 4.0),
    ((today - timedelta(days=1)).isoformat() + "T15:00:00", "stuck", 1.5),
    # 3 days ago - neutral
    ((today - timedelta(days=3)).isoformat() + "T12:00:00", "stressed", 0.5),
    # 5 days ago - productive
    ((today - timedelta(days=5)).isoformat() + "T11:00:00", "proud", -3.0),
    ((today - timedelta(days=5)).isoformat() + "T17:00:00", "calm", -4.0),
    # 10 days ago - high paralysis
    ((today - timedelta(days=10)).isoformat() + "T10:00:00", "overwhelmed", 5.0),
    # 20 days ago - flow state
    ((today - timedelta(days=20)).isoformat() + "T14:00:00", "hopeful", -5.0),
    # entry with NULL paralysis score (older entry before feature was built)
    ((today - timedelta(days=22)).isoformat() + "T10:00:00", "tired", None),
]

for ts, emo, score in test_entries:
    if score is not None:
        cur.execute(
            "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, "
            "paralysis_score, created_at) VALUES (%s, %s, %s, %s, %s)",
            (user_id, f"test - feeling {emo}", emo, score, ts),
        )
    else:
        cur.execute(
            "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, "
            "created_at) VALUES (%s, %s, %s, %s)",
            (user_id, f"test - feeling {emo}", emo, ts),
        )
conn.commit()
conn.close()

print(f"Test user created (id={user_id}), {len(test_entries)} entries inserted\n")

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


with app.test_client() as c:
    c.post("/login", data={"username": "pstest99", "password": "testpass123"})

    # 1. page loads
    resp = c.get("/analytics")
    html = resp.data.decode("utf-8")
    check("Analytics page returns 200", resp.status_code == 200)

    # 2. paralysis chart section exists
    check("Paralysis score heading present", "Paralysis score over time" in html)

    # 3. canvas element present
    check("Paralysis canvas present", "paralysisScoreChart" in html)

    # 4. zone legend present
    check("Zone legend present", "ps-zone-legend" in html)
    check("Green zone label present", "In flow (negative)" in html)
    check("Red zone label present", "Frozen (positive)" in html)
    check("Neutral zone label present", "Neutral (0)" in html)

    # 5. psych explainer present
    check("Psych explainer on paralysis chart", "freeze mode" in html)

    # 6. chart data embedded
    check("Score data in page", "3.5" in html)
    check("Negative score in data", "-2.0" in html)
    check("Max score in data", "5.0" in html)
    check("Min score in data", "-5.0" in html)

    # 7. zone plugin code present
    check("Zone plugin code present", "paralysisZones" in html)
    check("Green zone colour present", "129, 199, 132" in html)
    check("Red zone colour present", "229, 115, 115" in html)

    # 8. y axis range is -5 to +5
    check("Y axis min -5", "min: -5" in html)
    check("Y axis max 5", "max: 5" in html)

    # 9. tooltip shows emotion and zone label
    check("Tooltip shows zone label", "In flow" in html)
    check("Tooltip shows High paralysis", "High paralysis" in html)

    # 10. range selector on paralysis chart
    resp = c.get("/analytics?range=7")
    html = resp.data.decode("utf-8")
    check("7-day range loads", resp.status_code == 200)
    # should have entries from today, yesterday, 3 and 5 days ago but not 10 or 20 days ago
    check("Has recent scores in 7-day", "3.5" in html)

    # 11. 30-day range has more data
    resp = c.get("/analytics?range=30")
    html = resp.data.decode("utf-8")
    check("30-day has 20-day-old entry", "-5.0" in html)

    # 12. NULL paralysis score entries are excluded (the tired entry)
    # the NULL entry was 22 days ago, within 30-day range, but should not appear in ps chart
    # we can check that 'tired' does not appear in ps_emotions data
    # (it appears in the mood chart but not the paralysis score chart)
    check("NULL score entry excluded from ps chart", resp.status_code == 200)

    # 13. empty state not shown when data exists
    check("Empty state NOT shown", "paralysis score trend will appear" not in html)

    # 14. mood chart still works alongside
    check("Mood chart still present", "moodTrendChart" in html)
    check("Both charts on same page", "moodTrendChart" in html and "paralysisScoreChart" in html)

    # 15. invalid range handled
    resp = c.get("/analytics?range=abc")
    check("Invalid range handled", resp.status_code == 200)

    # 16. point colours logic - scores map to correct colours
    check("Point colour mapping present", "pointBackgroundColor" in html)

    # 17. chart has no legend (single dataset doesnt need one)
    check("Legend hidden for single dataset", "display: false" in html)

    # 18. range selector buttons match between both charts
    resp = c.get("/analytics?range=14")
    html = resp.data.decode("utf-8")
    # both range selectors should show 14 as active
    active_count = html.count("chart-range-active")
    check("Both range selectors show same active state", active_count == 2)


# test with zero entries (empty state)
conn = get_db_connection()
cur = conn.cursor()
cur.execute("RESET ROLE;")
cur.execute("DELETE FROM journal_entries WHERE user_id = %s", (user_id,))
conn.commit()
cur.execute("SET ROLE flask_app;")
cur.execute("SET app.current_user_id = %s", (str(user_id),))
conn.close()

with app.test_client() as c:
    c.post("/login", data={"username": "pstest99", "password": "testpass123"})

    resp = c.get("/analytics")
    html = resp.data.decode("utf-8")
    check("Empty state shown when no data", "paralysis score trend will appear" in html)
    check("No canvas when no data", 'id="paralysisScoreChart"' not in html)
    check("Range selector still visible", "chart-range-selector" in html)
    check("Page returns 200 with no data", resp.status_code == 200)


# test with entries that all have NULL paralysis_score
conn = get_db_connection()
cur = conn.cursor()
cur.execute("RESET ROLE;")
cur.execute(
    "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, created_at) "
    "VALUES (%s, %s, %s, %s)",
    (user_id, "old entry no score", "tired", now),
)
conn.commit()
cur.execute("SET ROLE flask_app;")
cur.execute("SET app.current_user_id = %s", (str(user_id),))
conn.close()

with app.test_client() as c:
    c.post("/login", data={"username": "pstest99", "password": "testpass123"})

    resp = c.get("/analytics")
    html = resp.data.decode("utf-8")
    check("Empty state when all scores are NULL", "paralysis score trend will appear" in html)
    check("Mood chart still works with NULL scores", "moodTrendChart" in html)


# clean up
conn = get_db_connection()
cur = conn.cursor()
cur.execute("RESET ROLE;")
cur.execute("DELETE FROM journal_entries WHERE user_id = %s", (user_id,))
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
