"""
Security tests - CSRF, input sanitisation, SQL injection, user isolation.
Run: python -m pytest test_security.py -v
"""
import os
import uuid
import pytest

os.environ['FLASK_ENV'] = 'development'

from app import app, get_db_connection, limiter, sanitize_input
from werkzeug.security import generate_password_hash
from datetime import datetime


@pytest.fixture
def client():
    return app.test_client()


@pytest.fixture(autouse=True)
def set_testing_mode():
    """keep CSRF ON for csrf tests, disable rate limiter"""
    app.config['TESTING'] = True
    limiter.enabled = False
    yield
    limiter.enabled = True


def unique_username():
    return f"sectest_{uuid.uuid4().hex[:8]}"


def create_onboarded_user(username, password="testpass123"):
    """create a user with onboarding done"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("RESET ROLE;")
    pw_hash = generate_password_hash(password)
    now = datetime.now().isoformat(timespec="seconds")
    cur.execute(
        "INSERT INTO users (username, password_hash, created_at, onboarding_complete) "
        "VALUES (%s, %s, %s, 1)",
        (username, pw_hash, now),
    )
    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
    user_id = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO alignment_state (user_id, alignment_score, emotional_streak, last_journal_date) "
        "VALUES (%s, 0, 0, NULL)",
        (user_id,),
    )
    cur.execute(
        "INSERT INTO goals (user_id, goal_text, created_at) VALUES (%s, %s, %s)",
        (user_id, "test goal", now),
    )
    cur.execute("SELECT id FROM goals WHERE user_id = %s LIMIT 1", (user_id,))
    goal_id = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO identity_beliefs (user_id, belief_text, linked_goal_id, created_at) "
        "VALUES (%s, %s, %s, %s)",
        (user_id, "I am focused", goal_id, now),
    )
    conn.commit()
    conn.close()
    return user_id


def login_user(client, username, password="testpass123"):
    # need CSRF disabled for login
    app.config['WTF_CSRF_ENABLED'] = False
    client.post("/login", data={"username": username, "password": password})
    app.config['WTF_CSRF_ENABLED'] = True


def cleanup_user(username):
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


# ---- CSRF Protection ----

class TestCSRFProtection:
    def test_journal_post_without_csrf_rejected(self, client):
        """POST to /journal without CSRF token should be rejected"""
        uname = unique_username()
        create_onboarded_user(uname)
        login_user(client, uname)

        # csrf is enabled, posting without token should fail
        resp = client.post("/journal", data={
            "entry_text": "test entry",
        }, follow_redirects=True)
        # should get 400 (csrf error handler) or the csrf error template
        assert resp.status_code == 400 or b"csrf" in resp.data.lower() or b"expired" in resp.data.lower()
        cleanup_user(uname)

    def test_login_post_without_csrf_rejected(self, client):
        """POST to /login without CSRF token should be rejected"""
        resp = client.post("/login", data={
            "username": "test",
            "password": "test",
        }, follow_redirects=True)
        assert resp.status_code == 400 or b"csrf" in resp.data.lower() or b"expired" in resp.data.lower()

    def test_signup_post_without_csrf_rejected(self, client):
        """POST to /signup without CSRF token should be rejected"""
        resp = client.post("/signup", data={
            "username": "test",
            "password": "test123",
            "confirm_password": "test123",
        }, follow_redirects=True)
        assert resp.status_code == 400 or b"csrf" in resp.data.lower() or b"expired" in resp.data.lower()


# ---- Input Sanitisation ----

class TestInputSanitisation:
    def test_sanitize_strips_script_tags(self):
        """sanitize_input should remove script tags"""
        result = sanitize_input("<script>alert('xss')</script>hello")
        assert "<script>" not in result
        assert "hello" in result

    def test_sanitize_strips_html_tags(self):
        """sanitize_input should strip all HTML"""
        result = sanitize_input("<b>bold</b> <i>italic</i>")
        assert "<b>" not in result
        assert "<i>" not in result
        assert "bold" in result

    def test_sanitize_handles_none(self):
        """sanitize_input should handle None gracefully"""
        result = sanitize_input(None)
        assert result is None

    def test_sanitize_handles_empty(self):
        """sanitize_input should handle empty string"""
        result = sanitize_input("")
        assert result == ""

    def test_journal_xss_stripped(self, client):
        """XSS in journal entry should be cleaned"""
        uname = unique_username()
        uid = create_onboarded_user(uname)
        login_user(client, uname)

        app.config['WTF_CSRF_ENABLED'] = False
        client.post("/journal", data={
            "entry_text": "<script>alert('xss')</script>I feel calm",
        })
        app.config['WTF_CSRF_ENABLED'] = True

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute(
            "SELECT entry_text FROM journal_entries WHERE user_id = %s ORDER BY id DESC LIMIT 1",
            (uid,),
        )
        entry = cur.fetchone()
        conn.close()
        if entry:
            assert "<script>" not in entry["entry_text"]
        cleanup_user(uname)


# ---- SQL Injection ----

class TestSQLInjection:
    def test_sql_injection_in_journal(self, client):
        """SQL injection attempt should be safely stored as text"""
        uname = unique_username()
        uid = create_onboarded_user(uname)
        login_user(client, uname)

        injection = "'; DROP TABLE journal_entries; --"
        app.config['WTF_CSRF_ENABLED'] = False
        resp = client.post("/journal", data={
            "entry_text": injection,
        }, follow_redirects=True)
        app.config['WTF_CSRF_ENABLED'] = True

        # should not crash
        assert resp.status_code == 200

        # table should still exist
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute("SELECT COUNT(*) as cnt FROM journal_entries")
        count = cur.fetchone()["cnt"]
        conn.close()
        assert count >= 0  # table exists and is queryable
        cleanup_user(uname)

    def test_sql_injection_in_login(self, client):
        """SQL injection in login should fail normally"""
        app.config['WTF_CSRF_ENABLED'] = False
        resp = client.post("/login", data={
            "username": "' OR 1=1 --",
            "password": "anything",
        }, follow_redirects=True)
        app.config['WTF_CSRF_ENABLED'] = True
        # should not crash and should not log in
        assert resp.status_code == 200
        assert b"invalid" in resp.data.lower() or b"error" in resp.data.lower()


# ---- User Isolation (Row Level Security) ----

class TestUserIsolation:
    def test_user_a_cannot_see_user_b_journal(self, client):
        """user A's journal entries should not be visible to user B"""
        uname_a = unique_username()
        uname_b = unique_username()
        uid_a = create_onboarded_user(uname_a)
        uid_b = create_onboarded_user(uname_b)

        # user A creates a journal entry
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        now = datetime.now().isoformat(timespec="seconds")
        cur.execute(
            "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, created_at) "
            "VALUES (%s, %s, %s, %s)",
            (uid_a, "secret entry from user A", "calm", now),
        )
        conn.commit()
        conn.close()

        # log in as user B and check journal page
        app.config['WTF_CSRF_ENABLED'] = False
        login_user(client, uname_b)
        resp = client.get("/journal")
        app.config['WTF_CSRF_ENABLED'] = True

        assert b"secret entry from user A" not in resp.data
        cleanup_user(uname_a)
        cleanup_user(uname_b)

    def test_user_a_cannot_see_user_b_habits(self, client):
        """user A's habits should not appear for user B"""
        uname_a = unique_username()
        uname_b = unique_username()
        uid_a = create_onboarded_user(uname_a)
        uid_b = create_onboarded_user(uname_b)

        # user A creates a custom habit
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        now = datetime.now().isoformat(timespec="seconds")
        cur.execute(
            "INSERT INTO habits (user_id, name, is_sample, created_at) "
            "VALUES (%s, %s, 0, %s)",
            (uid_a, "user A private habit xyz", now),
        )
        conn.commit()
        conn.close()

        # log in as user B and check habits
        app.config['WTF_CSRF_ENABLED'] = False
        login_user(client, uname_b)
        resp = client.get("/habits")
        app.config['WTF_CSRF_ENABLED'] = True

        assert b"user A private habit xyz" not in resp.data
        cleanup_user(uname_a)
        cleanup_user(uname_b)


# ---- Rate Limiting ----

class TestRateLimiting:
    def test_login_rate_limited(self, client):
        """hitting /login too many times should get rate limited"""
        # re-enable rate limiter for this test
        limiter.enabled = True
        app.config['WTF_CSRF_ENABLED'] = False

        got_429 = False
        for i in range(15):
            resp = client.post("/login", data={
                "username": "nobody",
                "password": "wrong",
            })
            if resp.status_code == 429:
                got_429 = True
                break

        app.config['WTF_CSRF_ENABLED'] = True
        limiter.enabled = False
        assert got_429, "Expected 429 rate limit response after rapid login attempts"
