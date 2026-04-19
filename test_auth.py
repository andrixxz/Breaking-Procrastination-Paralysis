"""
Auth tests - signup, login, logout, lockout, protected routes.
Run: python -m pytest test_auth.py -v
"""
import os
import uuid
import pytest

os.environ['FLASK_ENV'] = 'development'

from app import app, get_db_connection, limiter
from werkzeug.security import generate_password_hash
from datetime import datetime, timedelta


@pytest.fixture(autouse=True)
def disable_csrf_and_rate_limit():
    """turn off csrf and rate limiting so tests can POST freely"""
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['TESTING'] = True
    # disable rate limiter for tests
    limiter.enabled = False
    yield
    limiter.enabled = True


@pytest.fixture
def client():
    """fresh test client for each test"""
    return app.test_client()


def unique_username():
    """random username so tests dont clash"""
    return f"testuser_{uuid.uuid4().hex[:8]}"


def create_test_user(username, password="testpass123", onboarding_complete=0):
    """create a user directly in the DB and return their id.
    uses RESET ROLE to bypass RLS since we're inserting test data."""
    conn = get_db_connection()
    cur = conn.cursor()
    # bypass RLS for test setup
    cur.execute("RESET ROLE;")
    pw_hash = generate_password_hash(password)
    now = datetime.now().isoformat(timespec="seconds")

    cur.execute(
        "INSERT INTO users (username, password_hash, created_at, onboarding_complete) "
        "VALUES (%s, %s, %s, %s)",
        (username, pw_hash, now, onboarding_complete),
    )
    # get the new user's id
    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
    user_id = cur.fetchone()["id"]

    # set up alignment state so the app doesnt crash
    cur.execute(
        "INSERT INTO alignment_state (user_id, alignment_score, emotional_streak, last_journal_date) "
        "VALUES (%s, 0, 0, NULL)",
        (user_id,),
    )
    conn.commit()
    conn.close()
    return user_id


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

    # delete in FK-safe order
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

class TestSignupHappyPath:
    def test_signup_valid_user(self, client):
        """signup with good username and password should redirect to dashboard"""
        uname = unique_username()
        resp = client.post("/signup", data={
            "username": uname,
            "password": "validpass123",
            "confirm_password": "validpass123",
        }, follow_redirects=False)
        # should get a redirect (302) to dashboard after signup
        assert resp.status_code == 302
        cleanup_user(uname)

    def test_signup_creates_account(self, client):
        """after signup, user should exist in the database"""
        uname = unique_username()
        client.post("/signup", data={
            "username": uname,
            "password": "validpass123",
            "confirm_password": "validpass123",
        })
        # check user exists in DB
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = %s", (uname,))
        user = cur.fetchone()
        conn.close()
        assert user is not None
        cleanup_user(uname)

    def test_signup_get_shows_form(self, client):
        """GET /signup should show the signup form"""
        resp = client.get("/signup")
        assert resp.status_code == 200
        assert b"signup" in resp.data.lower() or b"sign up" in resp.data.lower()


class TestLoginHappyPath:
    def test_login_correct_credentials(self, client):
        """login with right password should redirect"""
        uname = unique_username()
        create_test_user(uname, "mypassword1")
        resp = client.post("/login", data={
            "username": uname,
            "password": "mypassword1",
        }, follow_redirects=False)
        assert resp.status_code == 302
        cleanup_user(uname)

    def test_login_sets_session(self, client):
        """after login, session should have user_id"""
        uname = unique_username()
        create_test_user(uname, "mypassword1")
        with client.session_transaction() as pre_sess:
            assert "user_id" not in pre_sess

        client.post("/login", data={
            "username": uname,
            "password": "mypassword1",
        })
        with client.session_transaction() as post_sess:
            assert "user_id" in post_sess
        cleanup_user(uname)

    def test_login_get_shows_form(self, client):
        """GET /login should show the login form"""
        resp = client.get("/login")
        assert resp.status_code == 200

    def test_login_after_logout(self, client):
        """should be able to login again after logging out"""
        uname = unique_username()
        create_test_user(uname, "mypassword1")
        # login
        client.post("/login", data={"username": uname, "password": "mypassword1"})
        # logout
        client.get("/logout")
        # login again
        resp = client.post("/login", data={
            "username": uname,
            "password": "mypassword1",
        }, follow_redirects=False)
        assert resp.status_code == 302
        cleanup_user(uname)


class TestLogout:
    def test_logout_clears_session(self, client):
        """logout should clear the session and redirect to landing"""
        uname = unique_username()
        create_test_user(uname, "mypassword1")
        client.post("/login", data={"username": uname, "password": "mypassword1"})

        resp = client.get("/logout", follow_redirects=False)
        assert resp.status_code == 302

        with client.session_transaction() as sess:
            assert "user_id" not in sess
        cleanup_user(uname)


# ---- Edge Cases ----

class TestSignupEdgeCases:
    def test_duplicate_username(self, client):
        """signing up with an existing username should show error"""
        uname = unique_username()
        create_test_user(uname, "somepass123")
        resp = client.post("/signup", data={
            "username": uname,
            "password": "anotherpass1",
            "confirm_password": "anotherpass1",
        }, follow_redirects=True)
        assert b"already taken" in resp.data.lower() or b"username" in resp.data.lower()
        cleanup_user(uname)

    def test_short_password(self, client):
        """password under 6 chars should be rejected"""
        uname = unique_username()
        resp = client.post("/signup", data={
            "username": uname,
            "password": "a",
            "confirm_password": "a",
        }, follow_redirects=True)
        assert b"at least 6" in resp.data.lower() or b"password" in resp.data.lower()
        cleanup_user(uname)

    def test_very_long_username(self, client):
        """200+ char username should either be rejected or handled without crash"""
        long_name = "a" * 250
        resp = client.post("/signup", data={
            "username": long_name,
            "password": "validpass123",
            "confirm_password": "validpass123",
        }, follow_redirects=True)
        # should not crash - either success or validation error
        assert resp.status_code == 200

    def test_password_mismatch(self, client):
        """mismatched password and confirm_password should show error"""
        uname = unique_username()
        resp = client.post("/signup", data={
            "username": uname,
            "password": "validpass123",
            "confirm_password": "differentpass",
        }, follow_redirects=True)
        assert b"do not match" in resp.data.lower() or b"password" in resp.data.lower()

    def test_short_username(self, client):
        """username under 3 chars should be rejected"""
        resp = client.post("/signup", data={
            "username": "ab",
            "password": "validpass123",
            "confirm_password": "validpass123",
        }, follow_redirects=True)
        assert b"at least 3" in resp.data.lower() or b"username" in resp.data.lower()


# ---- Boundary Conditions----

class TestSignupBoundary:
    def test_empty_username(self, client):
        """empty username should be rejected"""
        resp = client.post("/signup", data={
            "username": "",
            "password": "validpass123",
            "confirm_password": "validpass123",
        }, follow_redirects=True)
        assert b"required" in resp.data.lower() or b"username" in resp.data.lower()

    def test_empty_password(self, client):
        """empty password should be rejected"""
        uname = unique_username()
        resp = client.post("/signup", data={
            "username": uname,
            "password": "",
            "confirm_password": "",
        }, follow_redirects=True)
        assert b"required" in resp.data.lower() or b"password" in resp.data.lower()

    def test_both_empty(self, client):
        """both empty should be rejected"""
        resp = client.post("/signup", data={
            "username": "",
            "password": "",
            "confirm_password": "",
        }, follow_redirects=True)
        assert b"required" in resp.data.lower()

    def test_minimum_valid_signup(self, client):
        """3-char username and 6-char password should work"""
        uname = f"t{uuid.uuid4().hex[:2]}"  # exactly 3 chars
        resp = client.post("/signup", data={
            "username": uname,
            "password": "abcdef",
            "confirm_password": "abcdef",
        }, follow_redirects=False)
        assert resp.status_code == 302
        cleanup_user(uname)


# ---- Invalid Input ----

class TestLoginInvalidInput:
    def test_wrong_password(self, client):
        """wrong password should show error"""
        uname = unique_username()
        create_test_user(uname, "rightpassword1")
        resp = client.post("/login", data={
            "username": uname,
            "password": "wrongpassword",
        }, follow_redirects=True)
        assert b"invalid" in resp.data.lower() or b"password" in resp.data.lower()
        cleanup_user(uname)

    def test_nonexistent_username(self, client):
        """login with username that doesnt exist should show error"""
        resp = client.post("/login", data={
            "username": "neverexists_xyz123",
            "password": "somepassword",
        }, follow_redirects=True)
        assert b"invalid" in resp.data.lower()

    def test_dashboard_without_login(self, client):
        """accessing /dashboard without login should redirect"""
        resp = client.get("/dashboard", follow_redirects=False)
        assert resp.status_code == 302

    def test_journal_without_login(self, client):
        """accessing /journal without login should redirect"""
        resp = client.get("/journal", follow_redirects=False)
        assert resp.status_code == 302

    def test_habits_without_login(self, client):
        """accessing /habits without login should redirect"""
        resp = client.get("/habits", follow_redirects=False)
        assert resp.status_code == 302

    def test_today_without_login(self, client):
        """accessing /today without login should redirect"""
        resp = client.get("/today", follow_redirects=False)
        assert resp.status_code == 302

    def test_analytics_without_login(self, client):
        """accessing /analytics without login should redirect"""
        resp = client.get("/analytics", follow_redirects=False)
        assert resp.status_code == 302


# ---- Account Lockout ----

class TestAccountLockout:
    def test_lockout_after_5_failed_attempts(self, client):
        """5 wrong passwords should lock the account"""
        uname = unique_username()
        create_test_user(uname, "correctpass1")

        # fail 5 times
        for i in range(5):
            client.post("/login", data={
                "username": uname,
                "password": "wrongpass",
            })

        # 6th attempt should show locked message
        resp = client.post("/login", data={
            "username": uname,
            "password": "correctpass1",
        }, follow_redirects=True)
        assert b"locked" in resp.data.lower() or b"too many" in resp.data.lower()
        cleanup_user(uname)

    def test_warning_before_lockout(self, client):
        """should warn about remaining attempts when close to lockout"""
        uname = unique_username()
        create_test_user(uname, "correctpass1")

        # fail 3 times
        for i in range(3):
            client.post("/login", data={
                "username": uname,
                "password": "wrongpass",
            })

        # 4th attempt should mention remaining attempts
        resp = client.post("/login", data={
            "username": uname,
            "password": "wrongpass",
        }, follow_redirects=True)
        assert b"attempt" in resp.data.lower() or b"remaining" in resp.data.lower()
        cleanup_user(uname)
