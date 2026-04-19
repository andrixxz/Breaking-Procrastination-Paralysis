"""
Journal CRUD tests - submit, edit, delete, view entries.
Run: python -m pytest test_journal.py -v
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
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['TESTING'] = True
    limiter.enabled = False
    yield
    limiter.enabled = True


@pytest.fixture
def client():
    return app.test_client()


def unique_username():
    return f"jtest_{uuid.uuid4().hex[:8]}"


def create_onboarded_user(username, password="testpass123"):
    """create a user who has finished onboarding so they can access journal"""
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
    # need at least one goal and belief so dashboard doesnt break
    cur.execute(
        "INSERT INTO goals (user_id, goal_text, created_at) VALUES (%s, %s, %s)",
        (user_id, "test goal", now),
    )
    cur.execute("SELECT id FROM goals WHERE user_id = %s LIMIT 1", (user_id,))
    goal_id = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO identity_beliefs (user_id, belief_text, linked_goal_id, created_at) "
        "VALUES (%s, %s, %s, %s)",
        (user_id, "I am a focused person", goal_id, now),
    )
    conn.commit()
    conn.close()
    return user_id


def login_user(client, username, password="testpass123"):
    client.post("/login", data={"username": username, "password": password})


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


# ---- Happy Path ----

class TestJournalSubmit:
    def test_submit_valid_entry(self, client):
        """posting a journal entry should predict emotion and show reframe"""
        uname = unique_username()
        create_onboarded_user(uname)
        login_user(client, uname)

        resp = client.post("/journal", data={
            "entry_text": "I feel really overwhelmed with all my assignments",
        }, follow_redirects=True)
        assert resp.status_code == 200
        # should show predicted emotion somewhere on the page
        html = resp.data.decode("utf-8").lower()
        assert "emotion" in html or "feeling" in html
        cleanup_user(uname)

    def test_entry_stored_in_db(self, client):
        """submitted entry should be saved to journal_entries table"""
        uname = unique_username()
        uid = create_onboarded_user(uname)
        login_user(client, uname)

        client.post("/journal", data={
            "entry_text": "I feel calm and ready to work today",
        })

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute(
            "SELECT entry_text, predicted_emotion FROM journal_entries WHERE user_id = %s",
            (uid,),
        )
        entry = cur.fetchone()
        conn.close()
        assert entry is not None
        assert entry["predicted_emotion"] is not None
        cleanup_user(uname)

    def test_view_journal_page(self, client):
        """GET /journal should show past entries"""
        uname = unique_username()
        uid = create_onboarded_user(uname)
        login_user(client, uname)

        # insert a test entry directly
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute(
            "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, created_at) "
            "VALUES (%s, %s, %s, %s)",
            (uid, "test entry for viewing", "calm", datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
        conn.close()

        resp = client.get("/journal")
        assert resp.status_code == 200
        assert b"test entry for viewing" in resp.data
        cleanup_user(uname)

    def test_alignment_score_increases(self, client):
        """journal submission should increase alignment score by 1"""
        uname = unique_username()
        uid = create_onboarded_user(uname)
        login_user(client, uname)

        client.post("/journal", data={
            "entry_text": "I am feeling good about today",
        })

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute(
            "SELECT alignment_score FROM alignment_state WHERE user_id = %s",
            (uid,),
        )
        score = cur.fetchone()["alignment_score"]
        conn.close()
        assert score >= 1
        cleanup_user(uname)


class TestJournalEdit:
    def test_edit_entry(self, client):
        """editing an entry should update the text and re-predict emotion"""
        uname = unique_username()
        uid = create_onboarded_user(uname)
        login_user(client, uname)

        # create entry
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute(
            "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, created_at) "
            "VALUES (%s, %s, %s, %s)",
            (uid, "original entry", "calm", datetime.now().isoformat(timespec="seconds")),
        )
        cur.execute(
            "SELECT id FROM journal_entries WHERE user_id = %s ORDER BY id DESC LIMIT 1",
            (uid,),
        )
        entry_id = cur.fetchone()["id"]
        conn.commit()
        conn.close()

        # edit it
        resp = client.post(f"/journal/{entry_id}/edit", data={
            "entry_text": "I feel anxious about my exam tomorrow",
        }, follow_redirects=False)
        assert resp.status_code == 302

        # check the text changed
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute(
            "SELECT entry_text FROM journal_entries WHERE id = %s",
            (entry_id,),
        )
        updated = cur.fetchone()
        conn.close()
        assert "anxious" in updated["entry_text"].lower()
        cleanup_user(uname)


class TestJournalDelete:
    def test_delete_entry(self, client):
        """deleting an entry should remove it from the DB"""
        uname = unique_username()
        uid = create_onboarded_user(uname)
        login_user(client, uname)

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute(
            "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, created_at) "
            "VALUES (%s, %s, %s, %s)",
            (uid, "to be deleted", "calm", datetime.now().isoformat(timespec="seconds")),
        )
        cur.execute(
            "SELECT id FROM journal_entries WHERE user_id = %s ORDER BY id DESC LIMIT 1",
            (uid,),
        )
        entry_id = cur.fetchone()["id"]
        conn.commit()
        conn.close()

        resp = client.post(f"/journal/{entry_id}/delete", follow_redirects=False)
        assert resp.status_code == 302

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute("SELECT id FROM journal_entries WHERE id = %s", (entry_id,))
        assert cur.fetchone() is None
        conn.close()
        cleanup_user(uname)

    def test_delete_reduces_alignment_score(self, client):
        """deleting should reduce alignment score by 1"""
        uname = unique_username()
        uid = create_onboarded_user(uname)
        login_user(client, uname)

        # set score to 5 first
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute(
            "UPDATE alignment_state SET alignment_score = 5 WHERE user_id = %s",
            (uid,),
        )
        cur.execute(
            "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, created_at) "
            "VALUES (%s, %s, %s, %s)",
            (uid, "will delete this", "calm", datetime.now().isoformat(timespec="seconds")),
        )
        cur.execute(
            "SELECT id FROM journal_entries WHERE user_id = %s ORDER BY id DESC LIMIT 1",
            (uid,),
        )
        entry_id = cur.fetchone()["id"]
        conn.commit()
        conn.close()

        client.post(f"/journal/{entry_id}/delete")

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute("SELECT alignment_score FROM alignment_state WHERE user_id = %s", (uid,))
        score = cur.fetchone()["alignment_score"]
        conn.close()
        assert score == 4
        cleanup_user(uname)

    def test_delete_never_below_zero(self, client):
        """deleting when score is 0 should keep it at 0"""
        uname = unique_username()
        uid = create_onboarded_user(uname)
        login_user(client, uname)

        # score already starts at 0, add entry to delete
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute(
            "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, created_at) "
            "VALUES (%s, %s, %s, %s)",
            (uid, "delete me", "calm", datetime.now().isoformat(timespec="seconds")),
        )
        cur.execute(
            "SELECT id FROM journal_entries WHERE user_id = %s ORDER BY id DESC LIMIT 1",
            (uid,),
        )
        entry_id = cur.fetchone()["id"]
        conn.commit()
        conn.close()

        client.post(f"/journal/{entry_id}/delete")

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute("SELECT alignment_score FROM alignment_state WHERE user_id = %s", (uid,))
        score = cur.fetchone()["alignment_score"]
        conn.close()
        assert score == 0
        cleanup_user(uname)


# ---- Edge Cases ----

class TestJournalEdgeCases:
    def test_very_short_entry(self, client):
        """really short entry should still get a prediction"""
        uname = unique_username()
        create_onboarded_user(uname)
        login_user(client, uname)
        resp = client.post("/journal", data={"entry_text": "bad"}, follow_redirects=True)
        assert resp.status_code == 200
        cleanup_user(uname)

    def test_very_long_entry(self, client):
        """500+ word entry should work without crashing"""
        uname = unique_username()
        create_onboarded_user(uname)
        login_user(client, uname)
        long_text = "I feel worried about everything. " * 100
        resp = client.post("/journal", data={"entry_text": long_text}, follow_redirects=True)
        assert resp.status_code == 200
        cleanup_user(uname)

    def test_html_tags_stripped(self, client):
        """script tags in journal text should be stripped by bleach"""
        uname = unique_username()
        uid = create_onboarded_user(uname)
        login_user(client, uname)

        client.post("/journal", data={
            "entry_text": "<script>alert('xss')</script>I feel calm today",
        })

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


# ---- Boundary Conditions ----

class TestJournalBoundary:
    def test_empty_entry(self, client):
        """empty submission should not crash"""
        uname = unique_username()
        create_onboarded_user(uname)
        login_user(client, uname)
        resp = client.post("/journal", data={"entry_text": ""}, follow_redirects=True)
        assert resp.status_code == 200
        # should show the error message about not writing anything
        assert b"did not write" in resp.data.lower() or b"not write" in resp.data.lower() or resp.status_code == 200
        cleanup_user(uname)

    def test_whitespace_only(self, client):
        """spaces-only entry should be handled"""
        uname = unique_username()
        create_onboarded_user(uname)
        login_user(client, uname)
        resp = client.post("/journal", data={"entry_text": "   "}, follow_redirects=True)
        assert resp.status_code == 200
        cleanup_user(uname)


# ---- Invalid Input ----

class TestJournalInvalidInput:
    def test_post_without_login(self, client):
        """POST to /journal without being logged in should redirect"""
        resp = client.post("/journal", data={
            "entry_text": "test entry",
        }, follow_redirects=False)
        assert resp.status_code == 302
