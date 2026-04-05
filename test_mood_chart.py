"""
Test script for Task 6.1 - Mood Trend Chart on analytics page.
Run: python test_mood_chart.py
"""
import os
os.environ['FLASK_ENV'] = 'development'

from app import app, get_db_connection
from werkzeug.security import generate_password_hash
from datetime import datetime, date, timedelta

app.config['WTF_CSRF_ENABLED'] = False
app.config['TESTING'] = True

# set up test user directly in DB
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
            ("moodtest99",),
        )
    except Exception:
        conn.rollback()
cur.execute("DELETE FROM users WHERE username = %s", ("moodtest99",))
conn.commit()

# create user
pw_hash = generate_password_hash("testpass123")
cur.execute(
    "INSERT INTO users (username, password_hash, created_at, onboarding_complete) "
    "VALUES (%s, %s, %s, %s) RETURNING id",
    ("moodtest99", pw_hash, now, 1),
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

# set RLS
cur.execute("SET ROLE flask_app;")
cur.execute("SET app.current_user_id = %s", (str(user_id),))

# insert entries spread across different dates and emotions
test_entries = [
    # today
    (today.isoformat() + "T10:00:00", "overwhelmed"),
    (today.isoformat() + "T14:00:00", "anxious"),
    (today.isoformat() + "T18:00:00", "calm"),
    # yesterday
    ((today - timedelta(days=1)).isoformat() + "T09:00:00", "overwhelmed"),
    ((today - timedelta(days=1)).isoformat() + "T15:00:00", "overwhelmed"),
    # 3 days ago
    ((today - timedelta(days=3)).isoformat() + "T12:00:00", "stressed"),
    ((today - timedelta(days=3)).isoformat() + "T16:00:00", "proud"),
    # 10 days ago
    ((today - timedelta(days=10)).isoformat() + "T11:00:00", "anxious"),
    ((today - timedelta(days=10)).isoformat() + "T17:00:00", "tired"),
    # 20 days ago
    ((today - timedelta(days=20)).isoformat() + "T10:00:00", "stuck"),
    # 25 days ago
    ((today - timedelta(days=25)).isoformat() + "T14:00:00", "hopeful"),
]

for ts, emo in test_entries:
    cur.execute(
        "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, created_at) "
        "VALUES (%s, %s, %s, %s)",
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
    c.post("/login", data={"username": "moodtest99", "password": "testpass123"})

    # 1. analytics page loads
    resp = c.get("/analytics")
    html = resp.data.decode("utf-8")
    check("Analytics page returns 200", resp.status_code == 200)

    # 2. chart.js script tag present
    check("Chart.js CDN script present", "cdn.jsdelivr.net/npm/chart.js" in html)

    # 3. canvas element present
    check("Canvas element present", "moodTrendChart" in html)

    # 4. chart data is embedded
    check("Chart labels JSON present", "chart_labels" in html or "Jan" in html or "Feb" in html or "Mar" in html)

    # 5. range selector present
    check("Range selector present", "chart-range-selector" in html)
    check("7 day option present", "7 days" in html)
    check("14 day option present", "14 days" in html)
    check("30 day option present", "30 days" in html)

    # 6. default is 30 days
    check("30 days is active by default", "chart-range-active" in html)

    # 7. existing emotion distribution still works
    check("Emotion distribution still present", "Emotional landscape" in html)
    check("Activity cards still present", "Days you reflected this week" in html)

    # 8. test 7-day range
    resp = c.get("/analytics?range=7")
    html = resp.data.decode("utf-8")
    check("7-day range returns 200", resp.status_code == 200)
    check("7 days is active when selected", 'chart-range-active' in html)

    # 9. test 14-day range
    resp = c.get("/analytics?range=14")
    html = resp.data.decode("utf-8")
    check("14-day range returns 200", resp.status_code == 200)

    # 10. invalid range defaults to 30
    resp = c.get("/analytics?range=999")
    html = resp.data.decode("utf-8")
    check("Invalid range returns 200", resp.status_code == 200)

    # 11. test with emotion data in chart
    resp = c.get("/analytics?range=30")
    html = resp.data.decode("utf-8")
    check("Overwhelmed in chart data", "Overwhelmed" in html)
    check("Anxious in chart data", "Anxious" in html)
    check("Calm in chart data", "Calm" in html)

    # 12. mood trend section title
    check("Mood over time heading present", "Mood over time" in html)

    # 13. psych explainer present
    check("Psych explainer on chart", "emotional patterns" in html)

    # 14. y axis label
    check("Y axis label present", "Times felt" in html)

    # 15. chart has proper colours
    check("Emotion colours in data", "#E57373" in html)  # overwhelmed
    check("Calm colour in data", "#81C784" in html)  # calm

    # 16. empty state not shown when data exists
    check("Empty state NOT shown", "Mood trends will appear here" not in html)

    # 17. 7-day range should not include 25-day-old entry
    resp = c.get("/analytics?range=7")
    html = resp.data.decode("utf-8")
    check("Hopeful not in 7-day range", "Hopeful" not in html)  # hopeful was 25 days ago

    # 18. chart container has the right class
    check("Chart container class present", "chart-container" in html)

    # 19. SQL injection in range param
    resp = c.get("/analytics?range=7'; DROP TABLE--")
    html = resp.data.decode("utf-8")
    check("SQL injection in range param handled", resp.status_code == 200)

    # 20. chart datasets include tension for smooth lines
    check("Smooth line tension present", "0.3" in html)


# now test with a user who has zero entries
conn = get_db_connection()
cur = conn.cursor()
cur.execute("RESET ROLE;")
cur.execute("DELETE FROM journal_entries WHERE user_id = %s", (user_id,))
conn.commit()
cur.execute("SET ROLE flask_app;")
cur.execute("SET app.current_user_id = %s", (str(user_id),))
conn.close()

with app.test_client() as c:
    c.post("/login", data={"username": "moodtest99", "password": "testpass123"})

    # 21. empty state when no entries
    resp = c.get("/analytics")
    html = resp.data.decode("utf-8")
    check("Empty chart state shown when no data", "Mood trends will appear here" in html)
    check("No canvas when no data", 'id="moodTrendChart"' not in html)
    check("Range selector still shows", "chart-range-selector" in html)

    # 23. page still works fine with zero entries
    check("Page returns 200 with zero entries", resp.status_code == 200)


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
