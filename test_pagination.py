"""
Quick test script for journal pagination and filtering.
Run: python test_pagination.py
"""
import os
os.environ['FLASK_ENV'] = 'development'

from app import app, get_db_connection
from werkzeug.security import generate_password_hash
from datetime import datetime

app.config['WTF_CSRF_ENABLED'] = False
app.config['TESTING'] = True

# set up user directly in the DB to skip onboarding
conn = get_db_connection()
cur = conn.cursor()
now = datetime.now().isoformat(timespec="seconds")

# clean up if user exists from a previous run
cur.execute("RESET ROLE;")
# delete in FK-safe order
cur.execute("DELETE FROM habit_completions WHERE habit_id IN (SELECT id FROM habits WHERE user_id IN (SELECT id FROM users WHERE username = %s))", ("pagintest99",))
cur.execute("DELETE FROM step_completions WHERE step_id IN (SELECT id FROM goal_steps WHERE user_id IN (SELECT id FROM users WHERE username = %s))", ("pagintest99",))
cur.execute("DELETE FROM goal_steps WHERE user_id IN (SELECT id FROM users WHERE username = %s)", ("pagintest99",))
cur.execute("DELETE FROM positive_thoughts WHERE user_id IN (SELECT id FROM users WHERE username = %s)", ("pagintest99",))
cur.execute("DELETE FROM identity_beliefs WHERE user_id IN (SELECT id FROM users WHERE username = %s)", ("pagintest99",))
cur.execute("DELETE FROM todos WHERE user_id IN (SELECT id FROM users WHERE username = %s)", ("pagintest99",))
cur.execute("DELETE FROM habits WHERE user_id IN (SELECT id FROM users WHERE username = %s)", ("pagintest99",))
cur.execute("DELETE FROM journal_entries WHERE user_id IN (SELECT id FROM users WHERE username = %s)", ("pagintest99",))
cur.execute("DELETE FROM alignment_state WHERE user_id IN (SELECT id FROM users WHERE username = %s)", ("pagintest99",))
cur.execute("DELETE FROM goals WHERE user_id IN (SELECT id FROM users WHERE username = %s)", ("pagintest99",))
cur.execute("DELETE FROM users WHERE username = %s", ("pagintest99",))
conn.commit()

# create user with onboarding complete
pw_hash = generate_password_hash("testpass123")
cur.execute(
    "INSERT INTO users (username, password_hash, created_at, onboarding_complete) "
    "VALUES (%s, %s, %s, %s) RETURNING id",
    ("pagintest99", pw_hash, now, 1),
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

# set RLS for this user
cur.execute("SET ROLE flask_app;")
cur.execute("SET app.current_user_id = %s", (str(user_id),))

# insert 12 journal entries with different emotions
emotions = (
    ["overwhelmed"] * 8 + ["anxious"] * 2 + ["proud"] * 2
)
for i, emo in enumerate(emotions):
    cur.execute(
        "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, created_at) "
        "VALUES (%s, %s, %s, %s)",
        (user_id, f"test entry {i}", emo, now),
    )
conn.commit()
conn.close()

print(f"Test user created (id={user_id}), 12 entries inserted\n")

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
    # login
    c.post("/login", data={"username": "pagintest99", "password": "testpass123"})

    # 1. default page loads
    resp = c.get("/journal")
    html = resp.data.decode("utf-8")
    check("Journal page returns 200", resp.status_code == 200)
    check("Filter bar present", "journal-filter-bar" in html)
    check("Pagination controls present", "pagination-controls" in html)
    check("10 entries on default page", html.count("entry-item") == 10)
    check("Total count displayed", "12 reflections found" in html)

    # 2. per_page=5
    resp = c.get("/journal?per_page=5")
    html = resp.data.decode("utf-8")
    check("5 entries with per_page=5", html.count("entry-item") == 5)
    check("Page 2 link exists", "page=2" in html)
    check("Page 3 link exists", "page=3" in html)

    # 3. page 2
    resp = c.get("/journal?per_page=5&page=2")
    html = resp.data.decode("utf-8")
    check("5 entries on page 2", html.count("entry-item") == 5)
    check("Previous link on page 2", ">Previous<" in html)

    # 4. page 3 (last page: 12 entries / 5 per page = 2 remaining)
    resp = c.get("/journal?per_page=5&page=3")
    html = resp.data.decode("utf-8")
    check("2 entries on page 3", html.count("entry-item") == 2)

    # 5. per_page=25 (all fit)
    resp = c.get("/journal?per_page=25")
    html = resp.data.decode("utf-8")
    check("12 entries with per_page=25", html.count("entry-item") == 12)
    check("No pagination with per_page=25", "pagination-controls" not in html)

    # 6. emotion filter
    resp = c.get("/journal?emotion=overwhelmed")
    html = resp.data.decode("utf-8")
    check("Overwhelmed filter returns 200", resp.status_code == 200)
    check("8 overwhelmed entries", html.count("entry-item") == 8)

    # 7. emotion filter - proud (2 entries)
    resp = c.get("/journal?emotion=proud")
    html = resp.data.decode("utf-8")
    check("2 proud entries", html.count("entry-item") == 2)

    # 8. emotion filter - no results
    resp = c.get("/journal?emotion=calm")
    html = resp.data.decode("utf-8")
    check("No results message shown", "No reflections match" in html)

    # 9. invalid emotion ignored
    resp = c.get("/journal?emotion=fakemotion")
    html = resp.data.decode("utf-8")
    check("Invalid emotion ignored, shows all", html.count("entry-item") == 10)

    # 10. invalid per_page defaults to 10
    resp = c.get("/journal?per_page=9999")
    html = resp.data.decode("utf-8")
    check("Invalid per_page defaults to 10", html.count("entry-item") == 10)

    # 11. page beyond max clamped
    resp = c.get("/journal?page=999")
    html = resp.data.decode("utf-8")
    check("Page beyond max still returns 200", resp.status_code == 200)
    check("Page beyond max shows entries", html.count("entry-item") > 0)

    # 12. negative page clamped to 1
    resp = c.get("/journal?page=-5")
    html = resp.data.decode("utf-8")
    check("Negative page returns 200", resp.status_code == 200)

    # 13. week filter
    resp = c.get("/journal?period=week")
    html = resp.data.decode("utf-8")
    check("Week filter returns 200", resp.status_code == 200)
    check("Week filter shows entries", html.count("entry-item") > 0)

    # 14. month filter
    resp = c.get("/journal?period=month")
    html = resp.data.decode("utf-8")
    check("Month filter returns 200", resp.status_code == 200)

    # 15. custom date range
    resp = c.get("/journal?period=custom&date_from=2026-03-01&date_to=2026-03-31")
    html = resp.data.decode("utf-8")
    check("Custom date range returns 200", resp.status_code == 200)
    check("Custom range shows entries", html.count("entry-item") > 0)

    # 16. bad custom dates (ignored, shows all)
    resp = c.get("/journal?period=custom&date_from=DROP_TABLE&date_to=bad")
    html = resp.data.decode("utf-8")
    check("Bad dates dont crash", resp.status_code == 200)

    # 17. clear filters link when filtered
    resp = c.get("/journal?emotion=overwhelmed")
    html = resp.data.decode("utf-8")
    check("Clear filters link present when filtered", "Clear filters" in html)

    # 18. no clear filters when unfiltered
    resp = c.get("/journal")
    html = resp.data.decode("utf-8")
    check("No clear filters when unfiltered", "filter-clear" not in html)

    # 19. combined filters
    resp = c.get("/journal?emotion=overwhelmed&per_page=5")
    html = resp.data.decode("utf-8")
    check("Combined filters work", html.count("entry-item") == 5)

    # 20. SQL injection attempt
    resp = c.get("/journal?emotion=anxious' OR 1=1--")
    html = resp.data.decode("utf-8")
    check("SQL injection rejected (invalid emotion)", resp.status_code == 200)

    # 21. date range with only from
    resp = c.get("/journal?period=custom&date_from=2026-01-01")
    html = resp.data.decode("utf-8")
    check("Custom with only from date works", resp.status_code == 200)

    # 22. date range with only to
    resp = c.get("/journal?period=custom&date_to=2026-12-31")
    html = resp.data.decode("utf-8")
    check("Custom with only to date works", resp.status_code == 200)

    # 23. filter preserves in pagination links
    resp = c.get("/journal?emotion=overwhelmed&per_page=5")
    html = resp.data.decode("utf-8")
    check("Pagination preserves emotion filter", "emotion=overwhelmed" in html)

    # 24. per_page selector shows correct value
    resp = c.get("/journal?per_page=5")
    html = resp.data.decode("utf-8")
    check("Per page 5 selected in dropdown", 'value="5" selected' in html)

    # 25. emotion dropdown shows selected value
    resp = c.get("/journal?emotion=anxious")
    html = resp.data.decode("utf-8")
    check("Anxious selected in dropdown", 'value="anxious" selected' in html)


# clean up test user
conn = get_db_connection()
cur = conn.cursor()
cur.execute("RESET ROLE;")
cur.execute("DELETE FROM habit_completions WHERE habit_id IN (SELECT id FROM habits WHERE user_id = %s)", (user_id,))
cur.execute("DELETE FROM habits WHERE user_id = %s", (user_id,))
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
