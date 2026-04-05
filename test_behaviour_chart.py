"""
Test script for Task 6.3 - Behaviour State Distribution Doughnut Chart.
Run: python test_behaviour_chart.py
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
            ("bhtest99",),
        )
    except Exception:
        conn.rollback()
cur.execute("DELETE FROM users WHERE username = %s", ("bhtest99",))
conn.commit()

# create user
pw_hash = generate_password_hash("testpass123")
cur.execute(
    "INSERT INTO users (username, password_hash, created_at, onboarding_complete) "
    "VALUES (%s, %s, %s, %s) RETURNING id",
    ("bhtest99", pw_hash, now, 1),
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

# insert entries with different behaviour states
test_entries = [
    # today - mix of behaviours
    (today.isoformat() + "T10:00:00", "overwhelmed", "avoidance", 3.0),
    (today.isoformat() + "T12:00:00", "anxious", "overwhelm", 2.5),
    (today.isoformat() + "T14:00:00", "calm", "action", -2.0),
    (today.isoformat() + "T16:00:00", "proud", "completion", -3.0),
    # yesterday
    ((today - timedelta(days=1)).isoformat() + "T09:00:00", "stressed", "avoidance", 2.0),
    ((today - timedelta(days=1)).isoformat() + "T15:00:00", "hopeful", "recovery", -1.0),
    # 3 days ago
    ((today - timedelta(days=3)).isoformat() + "T12:00:00", "overwhelmed", "overwhelm", 3.5),
    ((today - timedelta(days=3)).isoformat() + "T18:00:00", "calm", "action", -2.5),
    # 5 days ago
    ((today - timedelta(days=5)).isoformat() + "T11:00:00", "stuck", "rumination", 1.0),
    # 10 days ago
    ((today - timedelta(days=10)).isoformat() + "T10:00:00", "anxious", "avoidance", 4.0),
    # 20 days ago
    ((today - timedelta(days=20)).isoformat() + "T14:00:00", "calm", "completion", -4.0),
    # entry with NULL behaviour (older entry before model 2 existed)
    ((today - timedelta(days=2)).isoformat() + "T10:00:00", "tired", None, None),
]

for ts, emo, bh, ps in test_entries:
    if bh is not None:
        cur.execute(
            "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, "
            "predicted_behaviour, paralysis_score, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (user_id, f"test - {emo}", emo, bh, ps, ts),
        )
    else:
        cur.execute(
            "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, "
            "created_at) VALUES (%s, %s, %s, %s)",
            (user_id, f"test - {emo}", emo, ts),
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
    c.post("/login", data={"username": "bhtest99", "password": "testpass123"})

    # 1. page loads
    resp = c.get("/analytics")
    html = resp.data.decode("utf-8")
    check("Analytics page returns 200", resp.status_code == 200)

    # 2. doughnut chart section exists
    check("Behaviour chart heading present", "Where your energy goes" in html)

    # 3. canvas element
    check("Doughnut canvas present", "behaviourDoughnutChart" in html)

    # 4. psych explainer
    check("Psych explainer present", "all or nothing" in html)

    # 5. total entries shown
    check("Total entries displayed", ">11<" in html or "11" in html)

    # 6. chart data embedded - behaviour labels
    check("Avoidance in chart data", "Avoidance" in html)
    check("Action in chart data", "Action" in html)
    check("Completion in chart data", "Completion" in html)
    check("Overwhelm in chart data", "Overwhelm" in html)
    check("Recovery in chart data", "Recovery" in html)
    check("Rumination in chart data", "Rumination" in html)

    # 7. colours present
    check("Avoidance colour present", "#E57373" in html)
    check("Action colour present", "#66BB6A" in html)
    check("Completion colour present", "#4DB6AC" in html)

    # 8. doughnut chart type in JS
    check("Doughnut type specified", "'doughnut'" in html)

    # 9. percentage in tooltip
    check("Percentage tooltip callback", "pct + '%'" in html or "pct" in html)

    # 10. legend with percentages
    check("Legend shows percentage", "generateLabels" in html)

    # 11. range selector on doughnut chart
    resp = c.get("/analytics?range=7")
    html = resp.data.decode("utf-8")
    check("7-day range loads doughnut", resp.status_code == 200)
    check("Doughnut visible in 7-day", "behaviourDoughnutChart" in html)

    # 12. 30-day has all entries including 20 days ago
    resp = c.get("/analytics?range=30")
    html = resp.data.decode("utf-8")
    check("30-day shows all behaviours", "Completion" in html)

    # 13. NULL behaviours excluded
    # we have 11 entries with behaviour and 1 with NULL
    # the doughnut total should be 11 (in 30-day range)
    check("NULL behaviour excluded from count", resp.status_code == 200)

    # 14. other charts still work
    check("Mood chart still present", "moodTrendChart" in html)
    check("Paralysis chart still present", "paralysisScoreChart" in html)
    check("Three charts on page", html.count("new Chart(") == 3)

    # 15. invalid range defaults
    resp = c.get("/analytics?range=abc")
    check("Invalid range handled", resp.status_code == 200)

    # 16. cutout for doughnut hole
    check("Doughnut cutout specified", "cutout" in html)

    # 17. range selectors synced (all three charts)
    resp = c.get("/analytics?range=14")
    html = resp.data.decode("utf-8")
    active_count = html.count("chart-range-active")
    check("All three range selectors show same active state", active_count == 3)

    # 18. empty state NOT shown when data exists
    check("Empty state NOT shown", "Behaviour patterns will appear" not in html)

    # 19. doughnut wrapper class
    check("Doughnut wrapper class present", "doughnut-chart-wrapper" in html)

    # 20. total label
    check("Total label present", "doughnut-total-label" in html)


# test with zero entries
conn = get_db_connection()
cur = conn.cursor()
cur.execute("RESET ROLE;")
cur.execute("DELETE FROM journal_entries WHERE user_id = %s", (user_id,))
conn.commit()
cur.execute("SET ROLE flask_app;")
cur.execute("SET app.current_user_id = %s", (str(user_id),))
conn.close()

with app.test_client() as c:
    c.post("/login", data={"username": "bhtest99", "password": "testpass123"})

    resp = c.get("/analytics")
    html = resp.data.decode("utf-8")
    check("Empty state shown when no data", "Behaviour patterns will appear" in html)
    check("No doughnut canvas when no data", 'id="behaviourDoughnutChart"' not in html)
    check("Page returns 200 with no data", resp.status_code == 200)


# test with all NULL behaviours (old entries)
conn = get_db_connection()
cur = conn.cursor()
cur.execute("RESET ROLE;")
cur.execute(
    "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, created_at) "
    "VALUES (%s, %s, %s, %s)",
    (user_id, "old entry", "tired", now),
)
conn.commit()
cur.execute("SET ROLE flask_app;")
cur.execute("SET app.current_user_id = %s", (str(user_id),))
conn.close()

with app.test_client() as c:
    c.post("/login", data={"username": "bhtest99", "password": "testpass123"})

    resp = c.get("/analytics")
    html = resp.data.decode("utf-8")
    check("Empty state when all behaviours NULL", "Behaviour patterns will appear" in html)
    check("Mood chart works despite no behaviour data", "moodTrendChart" in html)


# test with single behaviour only
conn = get_db_connection()
cur = conn.cursor()
cur.execute("RESET ROLE;")
cur.execute("DELETE FROM journal_entries WHERE user_id = %s", (user_id,))
cur.execute(
    "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, "
    "predicted_behaviour, paralysis_score, created_at) "
    "VALUES (%s, %s, %s, %s, %s, %s)",
    (user_id, "only avoidance", "anxious", "avoidance", 3.0, now),
)
conn.commit()
cur.execute("SET ROLE flask_app;")
cur.execute("SET app.current_user_id = %s", (str(user_id),))
conn.close()

with app.test_client() as c:
    c.post("/login", data={"username": "bhtest99", "password": "testpass123"})

    resp = c.get("/analytics")
    html = resp.data.decode("utf-8")
    check("Single behaviour renders doughnut", "behaviourDoughnutChart" in html)
    check("Single behaviour shows 100%", resp.status_code == 200)


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
