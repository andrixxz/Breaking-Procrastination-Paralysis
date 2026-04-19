"""
Habit and todo tests - add, toggle, delete, alignment score.
Run: python -m pytest test_habits_todos.py -v
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
    return f"httest_{uuid.uuid4().hex[:8]}"


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


def get_alignment_score(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("RESET ROLE;")
    cur.execute("SELECT alignment_score FROM alignment_state WHERE user_id = %s", (user_id,))
    score = cur.fetchone()["alignment_score"]
    conn.close()
    return score


# ---- Habits Happy Path ----

class TestHabitHappyPath:
    def test_add_custom_habit(self, client):
        """adding a habit should make it appear on the habits page"""
        uname = unique_username()
        uid = create_onboarded_user(uname)
        login_user(client, uname)

        resp = client.post("/habits", data={
            "new_habit": "Drink water every morning",
        }, follow_redirects=True)
        assert resp.status_code == 200
        assert b"Drink water every morning" in resp.data
        cleanup_user(uname)

    def test_toggle_habit_complete(self, client):
        """completing a habit should increase alignment score"""
        uname = unique_username()
        uid = create_onboarded_user(uname)
        login_user(client, uname)

        # add a habit directly
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        now = datetime.now().isoformat(timespec="seconds")
        cur.execute(
            "INSERT INTO habits (user_id, name, is_sample, created_at) "
            "VALUES (%s, %s, 0, %s)",
            (uid, "Test habit", now),
        )
        cur.execute(
            "SELECT id FROM habits WHERE user_id = %s AND name = %s",
            (uid, "Test habit"),
        )
        habit_id = cur.fetchone()["id"]
        conn.commit()
        conn.close()

        score_before = get_alignment_score(uid)
        client.post(f"/habit/{habit_id}/toggle")
        score_after = get_alignment_score(uid)

        assert score_after == score_before + 1
        cleanup_user(uname)

    def test_toggle_habit_uncomplete_no_score_decrease(self, client):
        """un-completing a habit should NOT decrease alignment score.
        This was a bug that was fixed - verify the fix holds."""
        uname = unique_username()
        uid = create_onboarded_user(uname)
        login_user(client, uname)

        # add and complete a habit
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        now = datetime.now().isoformat(timespec="seconds")
        cur.execute(
            "INSERT INTO habits (user_id, name, is_sample, created_at) "
            "VALUES (%s, %s, 0, %s)",
            (uid, "Toggle test", now),
        )
        cur.execute(
            "SELECT id FROM habits WHERE user_id = %s AND name = %s",
            (uid, "Toggle test"),
        )
        habit_id = cur.fetchone()["id"]
        conn.commit()
        conn.close()

        # complete it (score goes up)
        client.post(f"/habit/{habit_id}/toggle")
        score_after_complete = get_alignment_score(uid)

        # un-complete it (score should NOT go down)
        client.post(f"/habit/{habit_id}/toggle")
        score_after_uncomplete = get_alignment_score(uid)

        assert score_after_uncomplete == score_after_complete
        cleanup_user(uname)


# ---- To do Happy Path ----

class TestTodoHappyPath:
    def test_add_todo(self, client):
        """adding a todo should make it appear on the today page"""
        uname = unique_username()
        create_onboarded_user(uname)
        login_user(client, uname)

        resp = client.post("/todo/add-manual", data={
            "todo_text": "Buy groceries",
        }, follow_redirects=True)
        assert resp.status_code == 200
        assert b"Buy groceries" in resp.data
        cleanup_user(uname)

    def test_toggle_todo_complete(self, client):
        """completing a todo should increase alignment score"""
        uname = unique_username()
        uid = create_onboarded_user(uname)
        login_user(client, uname)

        # add a todo
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        now = datetime.now().isoformat(timespec="seconds")
        cur.execute(
            "INSERT INTO todos (user_id, text, source, is_done, created_at) "
            "VALUES (%s, %s, 'manual', 0, %s)",
            (uid, "Test todo", now),
        )
        cur.execute(
            "SELECT id FROM todos WHERE user_id = %s AND text = %s",
            (uid, "Test todo"),
        )
        todo_id = cur.fetchone()["id"]
        conn.commit()
        conn.close()

        score_before = get_alignment_score(uid)
        client.post(f"/todo/{todo_id}/toggle")
        score_after = get_alignment_score(uid)
        assert score_after == score_before + 1
        cleanup_user(uname)

    def test_toggle_todo_shows_win_message(self, client):
        """completing a todo should redirect with todo_done param for win card"""
        uname = unique_username()
        uid = create_onboarded_user(uname)
        login_user(client, uname)

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        now = datetime.now().isoformat(timespec="seconds")
        cur.execute(
            "INSERT INTO todos (user_id, text, source, is_done, created_at) "
            "VALUES (%s, %s, 'manual', 0, %s)",
            (uid, "Win test", now),
        )
        cur.execute(
            "SELECT id FROM todos WHERE user_id = %s AND text = %s",
            (uid, "Win test"),
        )
        todo_id = cur.fetchone()["id"]
        conn.commit()
        conn.close()

        resp = client.post(f"/todo/{todo_id}/toggle", follow_redirects=False)
        assert resp.status_code == 302
        assert "todo_done" in resp.headers.get("Location", "")
        cleanup_user(uname)

    def test_todo_uncomplete_no_score_increase(self, client):
        """un-completing a todo should NOT increase score (bug fix verification)"""
        uname = unique_username()
        uid = create_onboarded_user(uname)
        login_user(client, uname)

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        now = datetime.now().isoformat(timespec="seconds")
        cur.execute(
            "INSERT INTO todos (user_id, text, source, is_done, created_at) "
            "VALUES (%s, %s, 'manual', 0, %s)",
            (uid, "Toggle test todo", now),
        )
        cur.execute(
            "SELECT id FROM todos WHERE user_id = %s AND text = %s",
            (uid, "Toggle test todo"),
        )
        todo_id = cur.fetchone()["id"]
        conn.commit()
        conn.close()

        # complete
        client.post(f"/todo/{todo_id}/toggle")
        score_after_complete = get_alignment_score(uid)

        # un-complete
        client.post(f"/todo/{todo_id}/toggle")
        score_after_uncomplete = get_alignment_score(uid)

        assert score_after_uncomplete == score_after_complete
        cleanup_user(uname)


# ---- Edge Cases ----

class TestHabitTodoEdgeCases:
    def test_habit_with_long_text(self, client):
        """long habit name should not crash"""
        uname = unique_username()
        create_onboarded_user(uname)
        login_user(client, uname)
        long_name = "a" * 500
        resp = client.post("/habits", data={"new_habit": long_name}, follow_redirects=True)
        assert resp.status_code == 200
        cleanup_user(uname)

    def test_habit_with_html(self, client):
        """html in habit name should be stripped"""
        uname = unique_username()
        uid = create_onboarded_user(uname)
        login_user(client, uname)
        client.post("/habits", data={
            "new_habit": "<b>Bold habit</b><script>alert(1)</script>",
        })
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute(
            "SELECT name FROM habits WHERE user_id = %s AND is_sample = 0 ORDER BY id DESC LIMIT 1",
            (uid,),
        )
        habit = cur.fetchone()
        conn.close()
        if habit:
            assert "<script>" not in habit["name"]
            assert "<b>" not in habit["name"]
        cleanup_user(uname)

    def test_empty_habit_name(self, client):
        """empty habit name should not create a habit"""
        uname = unique_username()
        uid = create_onboarded_user(uname)
        login_user(client, uname)

        # count habits before
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute("SELECT COUNT(*) as cnt FROM habits WHERE user_id = %s AND is_sample = 0", (uid,))
        count_before = cur.fetchone()["cnt"]
        conn.close()

        client.post("/habits", data={"new_habit": ""})

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute("SELECT COUNT(*) as cnt FROM habits WHERE user_id = %s AND is_sample = 0", (uid,))
        count_after = cur.fetchone()["cnt"]
        conn.close()
        assert count_after == count_before
        cleanup_user(uname)

    def test_empty_todo_name(self, client):
        """empty todo should not be created"""
        uname = unique_username()
        uid = create_onboarded_user(uname)
        login_user(client, uname)

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute("SELECT COUNT(*) as cnt FROM todos WHERE user_id = %s", (uid,))
        count_before = cur.fetchone()["cnt"]
        conn.close()

        client.post("/todo/add-manual", data={"todo_text": ""})

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        cur.execute("SELECT COUNT(*) as cnt FROM todos WHERE user_id = %s", (uid,))
        count_after = cur.fetchone()["cnt"]
        conn.close()
        assert count_after == count_before
        cleanup_user(uname)


# ---- Boundary Conditions ----

class TestAlignmentScoreBoundary:
    def test_score_never_below_zero(self, client):
        """alignment score should never go negative"""
        uname = unique_username()
        uid = create_onboarded_user(uname)
        login_user(client, uname)

        # score starts at 0, add a todo and delete it
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("RESET ROLE;")
        now = datetime.now().isoformat(timespec="seconds")
        cur.execute(
            "INSERT INTO journal_entries (user_id, entry_text, predicted_emotion, created_at) "
            "VALUES (%s, %s, %s, %s)",
            (uid, "delete me", "calm", now),
        )
        cur.execute(
            "SELECT id FROM journal_entries WHERE user_id = %s ORDER BY id DESC LIMIT 1",
            (uid,),
        )
        entry_id = cur.fetchone()["id"]
        conn.commit()
        conn.close()

        # delete entry (which tries to reduce score by 1)
        client.post(f"/journal/{entry_id}/delete")

        score = get_alignment_score(uid)
        assert score >= 0
        cleanup_user(uname)
