"""
Onboarding flow tests - goals, beliefs, thoughts, habits, steps.
Run: python -m pytest test_onboarding.py -v
"""
import os
import uuid
import pytest

os.environ['FLASK_ENV'] = 'development'

from app import app, get_db_connection, limiter
from werkzeug.security import generate_password_hash
from datetime import datetime


@pytest.fixture(autouse=True)
def disable_csrf_and_rate_limit():
    """turn off csrf and rate limiting for tests"""
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['TESTING'] = True
    limiter.enabled = False
    yield
    limiter.enabled = True


@pytest.fixture
def client():
    return app.test_client()


def unique_username():
    return f"onbtest_{uuid.uuid4().hex[:8]}"


def create_user_no_onboarding(username, password="testpass123"):
    """create a user who hasnt done onboarding yet"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("RESET ROLE;")
    pw_hash = generate_password_hash(password)
    now = datetime.now().isoformat(timespec="seconds")
    cur.execute(
        "INSERT INTO users (username, password_hash, created_at, onboarding_complete) "
        "VALUES (%s, %s, %s, 0)",
        (username, pw_hash, now),
    )
    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
    user_id = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO alignment_state (user_id, alignment_score, emotional_streak, last_journal_date) "
        "VALUES (%s, 0, 0, NULL)",
        (user_id,),
    )
    conn.commit()
    conn.close()
    return user_id


def login_user(client, username, password="testpass123"):
    """helper to log a user in"""
    client.post("/login", data={"username": username, "password": password})


def cleanup_user(username):
    """remove test user and all their data"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("RESET ROLE;")
    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return
    uid = row["id"]
    cur.execute("DELETE FROM step_completions WHERE user_id = %s", (uid,))
    cur.execute("DELETE FROM goal_steps WHERE user_id = %s", (uid,))
    cur.execute("DELETE FROM habit_completions WHERE habit_id IN (SELECT id FROM habits WHERE user_id = %s)", (uid,))
    cur.execute("DELETE FROM positive_thoughts WHERE user_id = %s", (uid,))
    cur.execute("DELETE FROM identity_beliefs WHERE user_id = %s", (uid,))
    cur.execute("DELETE FROM todos WHERE user_id = %s", (uid,))
    cur.execute("DELETE FROM habits WHERE user_id = %s", (uid,))
    cur.execute("DELETE FROM journal_entries WHERE user_id = %s", (uid,))
    cur.execute("DELETE FROM alignment_state WHERE user_id = %s", (uid,))
    cur.execute("DELETE FROM goals WHERE user_id = %s", (uid,))
    cur.execute("DELETE FROM users WHERE id = %s", (uid,))
    conn.commit()
    conn.close()


# ---- Happy Path ----

class TestGoalsStep:
    def test_goals_page_loads(self, client):
        """GET /onboarding/goals should show the form"""
        uname = unique_username()
        create_user_no_onboarding(uname)
        login_user(client, uname)
        resp = client.get("/onboarding/goals")
        assert resp.status_code == 200
        cleanup_user(uname)

    def test_submit_3_valid_goals(self, client):
        """posting 3 goals should store them and redirect to beliefs"""
        uname = unique_username()
        uid = create_user_no_onboarding(uname)
        login_user(client, uname)
        resp = client.post("/onboarding/goals", data={
            "goal_1": "Finish my FYP",
            "goal_2": "Exercise 3 times a week",
            "goal_3": "Read more books",
        }, follow_redirects=False)
        assert resp.status_code == 302
        assert "/onboarding/beliefs" in resp.headers.get("Location", "")

        # check goals are in the DB
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute("SELECT goal_text FROM goals WHERE user_id = %s ORDER BY id", (uid,))
        goals = [r["goal_text"] for r in cur.fetchall()]
        conn.close()
        assert len(goals) == 3
        assert "Finish my FYP" in goals
        cleanup_user(uname)


class TestBeliefsStep:
    def test_submit_beliefs(self, client):
        """beliefs should be stored linked to goals"""
        uname = unique_username()
        uid = create_user_no_onboarding(uname)
        login_user(client, uname)

        # first do goals
        client.post("/onboarding/goals", data={
            "goal_1": "Goal A",
            "goal_2": "Goal B",
            "goal_3": "Goal C",
        })

        # get goal IDs
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute("SELECT id FROM goals WHERE user_id = %s ORDER BY id", (uid,))
        goal_ids = [r["id"] for r in cur.fetchall()]
        conn.close()

        # submit beliefs
        resp = client.post("/onboarding/beliefs", data={
            f"belief_{goal_ids[0]}": "I am a hard worker",
            f"belief_{goal_ids[1]}": "I am someone who moves",
            f"belief_{goal_ids[2]}": "I am a reader",
        }, follow_redirects=False)
        assert resp.status_code == 302

        # check beliefs stored
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute("SELECT belief_text FROM identity_beliefs WHERE user_id = %s", (uid,))
        beliefs = cur.fetchall()
        conn.close()
        assert len(beliefs) == 3
        cleanup_user(uname)


class TestThoughtsStep:
    def test_submit_thoughts(self, client):
        """thoughts should be stored linked to beliefs"""
        uname = unique_username()
        uid = create_user_no_onboarding(uname)
        login_user(client, uname)

        # goals first
        client.post("/onboarding/goals", data={
            "goal_1": "G1", "goal_2": "G2", "goal_3": "G3",
        })

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute("SELECT id FROM goals WHERE user_id = %s ORDER BY id", (uid,))
        goal_ids = [r["id"] for r in cur.fetchall()]
        conn.close()

        # beliefs
        client.post("/onboarding/beliefs", data={
            f"belief_{goal_ids[0]}": "B1",
            f"belief_{goal_ids[1]}": "B2",
            f"belief_{goal_ids[2]}": "B3",
        })

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute("SELECT id FROM identity_beliefs WHERE user_id = %s ORDER BY id", (uid,))
        belief_ids = [r["id"] for r in cur.fetchall()]
        conn.close()

        # thoughts
        resp = client.post("/onboarding/thoughts", data={
            f"thought_{belief_ids[0]}": "I can do this",
            f"thought_{belief_ids[1]}": "I feel strong",
            f"thought_{belief_ids[2]}": "I love learning",
        }, follow_redirects=False)
        assert resp.status_code == 302

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute("SELECT thought_text FROM positive_thoughts WHERE user_id = %s", (uid,))
        thoughts = cur.fetchall()
        conn.close()
        assert len(thoughts) == 3
        cleanup_user(uname)


class TestHabitsStep:
    def test_submit_habits(self, client):
        """habits should be stored linked to goals"""
        uname = unique_username()
        uid = create_user_no_onboarding(uname)
        login_user(client, uname)

        # run through goals, beliefs, thoughts first
        client.post("/onboarding/goals", data={
            "goal_1": "G1", "goal_2": "G2", "goal_3": "G3",
        })
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute("SELECT id FROM goals WHERE user_id = %s ORDER BY id", (uid,))
        goal_ids = [r["id"] for r in cur.fetchall()]
        conn.close()

        client.post("/onboarding/beliefs", data={
            f"belief_{goal_ids[0]}": "B1",
            f"belief_{goal_ids[1]}": "B2",
            f"belief_{goal_ids[2]}": "B3",
        })
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute("SELECT id FROM identity_beliefs WHERE user_id = %s ORDER BY id", (uid,))
        belief_ids = [r["id"] for r in cur.fetchall()]
        conn.close()

        client.post("/onboarding/thoughts", data={
            f"thought_{belief_ids[0]}": "T1",
            f"thought_{belief_ids[1]}": "T2",
            f"thought_{belief_ids[2]}": "T3",
        })

        # habits
        resp = client.post("/onboarding/habits", data={
            f"habit_{goal_ids[0]}": "Write 100 words",
            f"habit_{goal_ids[1]}": "Walk 10 minutes",
            f"habit_{goal_ids[2]}": "Read 5 pages",
        }, follow_redirects=False)
        assert resp.status_code == 302

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute(
            "SELECT name FROM habits WHERE user_id = %s AND is_sample = 0",
            (uid,),
        )
        habits = cur.fetchall()
        conn.close()
        assert len(habits) == 3
        cleanup_user(uname)


class TestFullOnboardingFlow:
    def test_end_to_end_onboarding(self, client):
        """complete all 5 steps - user should end up with onboarding_complete=1"""
        uname = unique_username()
        uid = create_user_no_onboarding(uname)
        login_user(client, uname)

        # step 1 - goals
        client.post("/onboarding/goals", data={
            "goal_1": "G1", "goal_2": "G2", "goal_3": "G3",
        })

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute("SELECT id FROM goals WHERE user_id = %s ORDER BY id", (uid,))
        goal_ids = [r["id"] for r in cur.fetchall()]
        conn.close()

        # step 2 - beliefs
        client.post("/onboarding/beliefs", data={
            f"belief_{goal_ids[0]}": "B1",
            f"belief_{goal_ids[1]}": "B2",
            f"belief_{goal_ids[2]}": "B3",
        })

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute("SELECT id FROM identity_beliefs WHERE user_id = %s ORDER BY id", (uid,))
        belief_ids = [r["id"] for r in cur.fetchall()]
        conn.close()

        # step 3 - thoughts
        client.post("/onboarding/thoughts", data={
            f"thought_{belief_ids[0]}": "T1",
            f"thought_{belief_ids[1]}": "T2",
            f"thought_{belief_ids[2]}": "T3",
        })

        # step 4 - habits
        client.post("/onboarding/habits", data={
            f"habit_{goal_ids[0]}": "H1",
            f"habit_{goal_ids[1]}": "H2",
            f"habit_{goal_ids[2]}": "H3",
        })

        # step 5 - steps (at least one per goal)
        resp = client.post("/onboarding/steps", data={
            f"step_{goal_ids[0]}_1": "First step for G1",
            f"freq_{goal_ids[0]}_1": "daily",
            f"step_{goal_ids[1]}_1": "First step for G2",
            f"freq_{goal_ids[1]}_1": "weekly",
            f"dow_{goal_ids[1]}_1": "1",
            f"step_{goal_ids[2]}_1": "First step for G3",
            f"freq_{goal_ids[2]}_1": "one-off",
        }, follow_redirects=False)
        assert resp.status_code == 302

        # verify onboarding is complete
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute("SELECT onboarding_complete FROM users WHERE id = %s", (uid,))
        user = cur.fetchone()
        conn.close()
        assert user["onboarding_complete"] == 1
        cleanup_user(uname)


# ---- Edge Cases ----

class TestOnboardingEdgeCases:
    def test_empty_goals(self, client):
        """submitting empty goals should show error.
        BUG FOUND: the route was missing existing_goals in the template context
        on validation failure. Fixed by passing [goal_1, goal_2, goal_3] to the
        render call."""
        uname = unique_username()
        create_user_no_onboarding(uname)
        login_user(client, uname)
        resp = client.post("/onboarding/goals", data={
            "goal_1": "",
            "goal_2": "",
            "goal_3": "",
        }, follow_redirects=True)
        assert resp.status_code == 200
        assert b"please" in resp.data.lower() or b"goal" in resp.data.lower()
        cleanup_user(uname)

    def test_skip_to_dashboard_before_onboarding(self, client):
        """trying to access dashboard before onboarding should redirect to onboarding"""
        uname = unique_username()
        create_user_no_onboarding(uname)
        login_user(client, uname)
        resp = client.get("/dashboard", follow_redirects=False)
        assert resp.status_code == 302
        location = resp.headers.get("Location", "")
        assert "onboarding" in location
        cleanup_user(uname)

    def test_beliefs_without_goals_first(self, client):
        """going to beliefs step without goals should redirect back"""
        uname = unique_username()
        create_user_no_onboarding(uname)
        login_user(client, uname)
        resp = client.get("/onboarding/beliefs", follow_redirects=False)
        assert resp.status_code == 302
        cleanup_user(uname)


# ---- Invalid Input ----

class TestOnboardingInvalidInput:
    def test_long_goal_text(self, client):
        """very long goal text should not crash the app"""
        uname = unique_username()
        create_user_no_onboarding(uname)
        login_user(client, uname)
        long_text = "a" * 1200
        resp = client.post("/onboarding/goals", data={
            "goal_1": long_text,
            "goal_2": "Normal goal",
            "goal_3": "Another goal",
        }, follow_redirects=True)
        # should not crash
        assert resp.status_code == 200 or resp.status_code == 302
        cleanup_user(uname)

    def test_html_in_goal_text(self, client):
        """html tags in goals should be stripped by bleach"""
        uname = unique_username()
        uid = create_user_no_onboarding(uname)
        login_user(client, uname)
        resp = client.post("/onboarding/goals", data={
            "goal_1": "<script>alert('xss')</script>Finish project",
            "goal_2": "Normal goal",
            "goal_3": "Another goal",
        }, follow_redirects=True)

        # check what actually got stored
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute(
            "SELECT goal_text FROM goals WHERE user_id = %s ORDER BY id ASC LIMIT 1",
            (uid,),
        )
        row = cur.fetchone()
        conn.close()
        if row:
            # script tags should be stripped
            assert "<script>" not in row["goal_text"]
        cleanup_user(uname)
