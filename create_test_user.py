"""
Creates a test user with lots of realistic data for demo/testing purposes.
Login: testuser / testpass123

Run with: python create_test_user.py
"""

import os
import sys
from datetime import datetime, date, timedelta
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv
import random

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
USE_POSTGRES = DATABASE_URL is not None

if USE_POSTGRES:
    import psycopg2
    import psycopg2.extras
    print("Using PostgreSQL (Supabase)")
else:
    import sqlite3
    print("Using SQLite (local dev)")

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "journal.db")


# same wrapper from app.py so queries work with both databases
class SQLiteCursorWrapper:
    def __init__(self, cursor):
        self._cursor = cursor
        self.lastrowid = None
        self.description = cursor.description

    def execute(self, query, params=None):
        query = query.replace('%s', '?')
        if params:
            self._cursor.execute(query, params)
        else:
            self._cursor.execute(query)
        self.lastrowid = self._cursor.lastrowid
        self.description = self._cursor.description
        return self

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()


class SQLiteConnectionWrapper:
    def __init__(self, conn):
        self._conn = conn

    def cursor(self):
        return SQLiteCursorWrapper(self._conn.cursor())

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()


def get_db_connection(rls_user_id=None):
    if USE_POSTGRES:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
        rls_cur = conn.cursor()
        rls_cur.execute("SET ROLE flask_app;")
        if rls_user_id is not None:
            rls_cur.execute("SET app.current_user_id = %s", (str(rls_user_id),))
        rls_cur.close()
        return conn
    else:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return SQLiteConnectionWrapper(conn)


def insert_and_get_id(cur, query, params):
    if USE_POSTGRES:
        cur.execute(query + " RETURNING id", params)
        return cur.fetchone()['id']
    else:
        cur.execute(query, params)
        return cur.lastrowid


def main():
    TEST_USERNAME = "Amber"
    TEST_PASSWORD = "testpass123"

    today = date.today()

    # check if user already exists
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = %s", (TEST_USERNAME,))
    existing = cur.fetchone()
    conn.close()

    if existing:
        uid = existing['id']
        print(f"User '{TEST_USERNAME}' already exists (id={uid}). Deleting and recreating...")
        # reconnect with the user's id so RLS allows the deletes
        conn = get_db_connection(rls_user_id=uid)
        cur = conn.cursor()
        # delete all data for this user in the right order (foreign keys)
        cur.execute("DELETE FROM step_completions WHERE user_id = %s", (uid,))
        cur.execute("DELETE FROM goal_steps WHERE user_id = %s", (uid,))
        cur.execute("DELETE FROM habit_completions WHERE habit_id IN (SELECT id FROM habits WHERE user_id = %s)", (uid,))
        cur.execute("DELETE FROM habits WHERE user_id = %s", (uid,))
        cur.execute("DELETE FROM positive_thoughts WHERE user_id = %s", (uid,))
        cur.execute("DELETE FROM identity_beliefs WHERE user_id = %s", (uid,))
        cur.execute("DELETE FROM todos WHERE user_id = %s", (uid,))
        cur.execute("DELETE FROM journal_entries WHERE user_id = %s", (uid,))
        cur.execute("DELETE FROM alignment_state WHERE user_id = %s", (uid,))
        cur.execute("DELETE FROM goals WHERE user_id = %s", (uid,))
        cur.execute("DELETE FROM users WHERE id = %s", (uid,))
        conn.commit()
        conn.close()

    # ---- 1. Create user ----
    conn = get_db_connection()
    cur = conn.cursor()
    password_hash = generate_password_hash(TEST_PASSWORD)
    created_at = (today - timedelta(days=28)).isoformat() + "T09:00:00"

    user_id = insert_and_get_id(
        cur,
        "INSERT INTO users (username, password_hash, created_at, onboarding_complete) VALUES (%s, %s, %s, 1)",
        (TEST_USERNAME, password_hash, created_at)
    )
    conn.commit()
    conn.close()
    print(f"Created user '{TEST_USERNAME}' with id={user_id}")

    # reconnect with RLS for this user
    conn = get_db_connection(rls_user_id=user_id)
    cur = conn.cursor()

    # ---- 2. Alignment state ----
    cur.execute(
        "INSERT INTO alignment_state (user_id, alignment_score, emotional_streak, last_journal_date) "
        "VALUES (%s, %s, %s, %s)",
        (user_id, 87, 5, (today - timedelta(days=0)).isoformat())
    )

    # ---- 3. Goals (3 intentions) ----
    goal_data = [
        "Finish my final year project and submit on time",
        "Build a consistent study routine instead of cramming",
        "Start exercising regularly even when I do not feel like it",
    ]
    goal_ids = []
    onboarding_time = (today - timedelta(days=28)).isoformat() + "T09:15:00"
    for g in goal_data:
        gid = insert_and_get_id(
            cur,
            "INSERT INTO goals (user_id, goal_text, created_at) VALUES (%s, %s, %s)",
            (user_id, g, onboarding_time)
        )
        goal_ids.append(gid)

    # ---- 4. Identity beliefs (linked to goals) ----
    belief_data = [
        (goal_ids[0], "I am someone who shows up and does the work, even when it feels hard"),
        (goal_ids[1], "I am someone who studies a little every day instead of waiting until the last minute"),
        (goal_ids[2], "I am someone who moves their body because it helps me think clearly"),
    ]
    belief_ids = []
    for gid, text in belief_data:
        bid = insert_and_get_id(
            cur,
            "INSERT INTO identity_beliefs (user_id, belief_text, linked_goal_id, created_at) VALUES (%s, %s, %s, %s)",
            (user_id, text, gid, onboarding_time)
        )
        belief_ids.append(bid)

    # ---- 5. Positive thoughts (linked to beliefs) ----
    thought_data = [
        (belief_ids[0], "I have already built something real. I can keep going."),
        (belief_ids[1], "Every small study session adds up to something bigger than I realise."),
        (belief_ids[2], "Even a 10 minute walk changes how I feel for the rest of the day."),
    ]
    for bid, text in thought_data:
        insert_and_get_id(
            cur,
            "INSERT INTO positive_thoughts (user_id, thought_text, linked_belief_id, created_at) VALUES (%s, %s, %s, %s)",
            (user_id, text, bid, onboarding_time)
        )

    # ---- 6. Habits (3 sample + 4 custom) ----
    sample_habits = [
        "Write one sentence for an assignment",
        "Open my notes and read for 5 minutes",
        "Tidy my desk for two minutes",
    ]
    sample_habit_ids = []
    for h in sample_habits:
        hid = insert_and_get_id(
            cur,
            "INSERT INTO habits (user_id, name, is_sample, created_at) VALUES (%s, %s, 1, %s)",
            (user_id, h, onboarding_time)
        )
        sample_habit_ids.append(hid)

    custom_habits = [
        ("Go for a 15 minute walk", goal_ids[2]),
        ("Review lecture notes for 10 minutes", goal_ids[1]),
        ("Work on FYP for 20 minutes", goal_ids[0]),
        ("Drink a glass of water before starting work", None),
    ]
    custom_habit_ids = []
    for h_name, linked_gid in custom_habits:
        hid = insert_and_get_id(
            cur,
            "INSERT INTO habits (user_id, name, is_sample, created_at, linked_goal_id) VALUES (%s, %s, 0, %s, %s)",
            (user_id, h_name, onboarding_time, linked_gid)
        )
        custom_habit_ids.append(hid)

    all_habit_ids = sample_habit_ids + custom_habit_ids

    # ---- 7. Habit completions (spread across 21 days) ----
    # simulate realistic pattern: misses some days, completes more as time goes on
    for days_ago in range(21, -1, -1):
        d = today - timedelta(days=days_ago)
        completion_time = d.isoformat() + "T14:30:00"

        # early days: complete 2-3 habits. recent days: complete 4-6
        if days_ago > 14:
            n_habits = random.randint(1, 3)
        elif days_ago > 7:
            n_habits = random.randint(2, 5)
        else:
            n_habits = random.randint(3, len(all_habit_ids))

        # skip some days entirely (realistic)
        if days_ago in [19, 15, 10, 6]:
            continue

        chosen = random.sample(all_habit_ids, min(n_habits, len(all_habit_ids)))
        for hid in chosen:
            insert_and_get_id(
                cur,
                "INSERT INTO habit_completions (habit_id, completed_at) VALUES (%s, %s)",
                (hid, completion_time)
            )

    # ---- 8. Goal steps (micro-steps for each goal) ----
    goal_steps_data = [
        # FYP goal
        (goal_ids[0], "Write the introduction chapter", 1, "one-off", (today + timedelta(days=3)).isoformat(), None),
        (goal_ids[0], "Finish the ML pipeline section of the report", 2, "one-off", (today + timedelta(days=5)).isoformat(), None),
        (goal_ids[0], "Prepare demo script and practice it", 3, "one-off", (today + timedelta(days=7)).isoformat(), None),
        (goal_ids[0], "Review and proofread the full report", 4, "one-off", (today + timedelta(days=10)).isoformat(), None),
        (goal_ids[0], "Code review and clean up app.py", 5, "one-off", (today - timedelta(days=2)).isoformat(), None),
        # Study routine goal
        (goal_ids[1], "Review notes from this week", 1, "weekly", None, 4),  # Thursday
        (goal_ids[1], "Read one research paper", 2, "weekly", None, 1),  # Monday
        (goal_ids[1], "Do practice problems for 15 minutes", 3, "daily", None, None),
        (goal_ids[1], "Summarise what I learned today in 3 sentences", 4, "daily", None, None),
        # Exercise goal
        (goal_ids[2], "Go for a morning walk", 1, "daily", None, None),
        (goal_ids[2], "Do a 10 minute stretch routine", 2, "daily", None, None),
        (goal_ids[2], "Try a new workout video", 3, "weekly", None, 5),  # Friday
    ]

    step_ids = []
    for gid, text, order, freq, due, dow in goal_steps_data:
        sid = insert_and_get_id(
            cur,
            "INSERT INTO goal_steps (goal_id, user_id, step_text, step_order, frequency, due_date, day_of_week, is_done, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, 0, %s)",
            (gid, user_id, text, order, freq, due, dow, onboarding_time)
        )
        step_ids.append(sid)

    # mark the one-off step that was due 2 days ago as done
    cur.execute("UPDATE goal_steps SET is_done = 1 WHERE id = %s", (step_ids[4],))

    # ---- 9. Step completions for daily/weekly steps (spread across days) ----
    daily_step_ids = [step_ids[7], step_ids[8], step_ids[9], step_ids[10]]  # the daily ones
    weekly_step_ids = [step_ids[5], step_ids[6], step_ids[11]]

    for days_ago in range(14, -1, -1):
        d = today - timedelta(days=days_ago)
        # skip some days
        if days_ago in [12, 9, 6]:
            continue
        # complete 1-3 daily steps each day
        n_daily = random.randint(1, min(3, len(daily_step_ids)))
        chosen_daily = random.sample(daily_step_ids, n_daily)
        for sid in chosen_daily:
            insert_and_get_id(
                cur,
                "INSERT INTO step_completions (step_id, user_id, completed_date) VALUES (%s, %s, %s)",
                (sid, user_id, d.isoformat())
            )
        # complete weekly steps on their matching days
        for sid in weekly_step_ids:
            idx = step_ids.index(sid)
            step_dow = goal_steps_data[idx][5]  # day_of_week
            if step_dow is not None and d.weekday() == step_dow:
                insert_and_get_id(
                    cur,
                    "INSERT INTO step_completions (step_id, user_id, completed_date) VALUES (%s, %s, %s)",
                    (sid, user_id, d.isoformat())
                )

    # also complete the one-off step that was marked done
    insert_and_get_id(
        cur,
        "INSERT INTO step_completions (step_id, user_id, completed_date) VALUES (%s, %s, %s)",
        (step_ids[4], user_id, (today - timedelta(days=2)).isoformat())
    )

    # ---- 10. Journal entries (45+ entries across 28 days) ----
    # realistic student journal data showing a journey from high paralysis to growing confidence
    journal_entries = [
        # Week 1 - struggling, high paralysis
        (-28, "09:30", "I have so much to do for my project and I just cannot figure out where to start. Every time I sit down I end up scrolling my phone instead.", "overwhelmed", "avoidance", 4.0,
         "Your brain is trying to protect you from something that feels too big. That makes sense. You said you are someone who shows up and does the work, even when it feels hard. Opening one file, just to look at it, is showing up.",
         "Open your project folder and look at one file for 30 seconds", 1),
        (-28, "16:00", "I tried to start but I just sat there staring at the screen for 20 minutes. I feel so useless.", "stuck", "avoidance", 3.5,
         "Your mind is looping because it is trying to solve everything at once. You do not need to. You said you are someone who shows up and does the work. Just opening the document was showing up. Taking this small step is one way to live that out.",
         "Write one sentence. Any sentence. It does not have to be good.", 2),
        (-27, "10:15", "I am so stressed about the deadline. I keep thinking about everything that could go wrong and it is making me feel sick.", "anxious", "rumination", 3.0,
         "Anxiety is your brain saying this matters to you. That is not a weakness. You said you are someone who shows up and does the work, even when it feels hard. One small action from here is how that becomes real.",
         "Write down the one smallest part of what is worrying you", 2),
        (-26, "11:00", "I wasted the entire morning doing nothing productive. I should have been working on my report but instead I watched videos. I always do this.", "guilty", "avoidance", 4.5,
         "Guilt does not help you start. Noticing it does. You noticed. That is enough. You said you are someone who shows up and does the work, even when it feels hard. One forward step right now is how that starts again.",
         "Name one thing you can do in the next 10 minutes. Just one.", 2),
        (-25, "14:00", "I am tired of being tired. I did not sleep well and now I cannot concentrate on anything. Everything feels like too much effort.", "tired", "avoidance", 2.0,
         "Tiredness is real and valid. You do not need to push through it. But you can do something very small. You said you are someone who shows up and does the work. Showing up even now is part of that.",
         "Read one paragraph of your notes. Just one.", 2),
        (-24, "09:00", "I feel like everyone else has their life together and I am the only one who cannot get started. My project is behind and I hate looking at it.", "frustrated", "avoidance", 3.5,
         "That frustration means you care about doing well. That is not nothing. You said you are someone who shows up and does the work, even when it feels hard. Channeling this energy into one action is how that becomes real.",
         "Open your project and type one thing. Anything. A comment, a variable name, a sentence.", 2),
        (-23, "12:00", "I managed to write a few paragraphs today. It is not great but at least I did something. Still feel stressed but slightly less than yesterday.", "stressed", "action", 0.5,
         "You wrote something. That is real progress. The stress might still be there but you moved through it. You said you are someone who shows up and does the work. What you are doing right now is proof of that.",
         "Keep going for another 5 minutes if you can", 5),

        # Week 2 - starting to show up more
        (-22, "10:00", "Actually sat down and coded for an hour today. I was nervous at first but once I started it was not that bad. Why do I always make it bigger in my head than it actually is.", "anxious", "action", -0.5,
         "You started even though you were nervous. That is exactly what breaking the pattern looks like. You said you are someone who shows up and does the work, even when it feels hard. What you are doing right now is proof of that.",
         "Take a short break and come back for another focused session", 5),
        (-21, "16:30", "Tried to study for my other module but I just could not focus. My mind kept drifting back to the FYP deadline. Ended up doing nothing for either.", "stuck", "rumination", 2.5,
         "Your mind is spinning because it is trying to solve everything at once. You do not need to solve both things right now. You said you are someone who studies a little every day. Moving from thinking to doing, even briefly, is how that becomes real.",
         "Set a 5 minute timer and read just one page of notes", 5),
        (-20, "09:45", "I am sitting here knowing I need to work but I just cannot bring myself to open the laptop. It is like there is a wall between me and the task.", "stuck", "avoidance", 3.0,
         "That wall is real. It is your nervous system trying to protect you. But you can walk around it by making the first step absurdly small. You said you are someone who shows up and does the work. Taking this small step is one way to live that out.",
         "Just open the laptop. You do not have to do anything else.", 1),
        (-20, "14:00", "Opened the laptop after journaling this morning and actually got some work done. The wall feeling went away after about 5 minutes of just being present with the work. Weird how that happens.", "calm", "action", -2.0,
         "You noticed something important. The resistance dissolves once you start. That is your prefrontal cortex re-engaging. You said you are someone who shows up and does the work, even when it feels hard. What you are doing right now is proof of that.",
         "Keep going gently. You are in a good space right now.", 5),
        (-19, "11:30", "Feeling unmotivated today. I know I should care but I just do not feel anything about the project right now. Like what is even the point.", "unmotivated", "avoidance", 2.5,
         "Not feeling motivated does not mean you cannot act. Motivation usually shows up after you start, not before. You said you are someone who shows up and does the work, even when it feels hard. Acting without motivation is the strongest form of that.",
         "Open your project and change one small thing. A comment, a colour, anything.", 2),
        (-18, "10:00", "Had a good conversation with my supervisor yesterday and now I actually feel a bit hopeful about the project. Maybe I can pull this off if I keep showing up.", "hopeful", "action", -2.5,
         "Notice that feeling. It came from showing up and doing the work. That is not luck. That is who you are becoming. You said you are someone who shows up and does the work. This feeling is evidence of that.",
         "Use this energy. Work on the next section while the momentum is here.", 10),
        (-17, "15:00", "I have been going back and forth on the same paragraph for an hour. I keep rewriting it and nothing sounds right. Going in circles.", "frustrated", "rumination", 2.0,
         "Perfectionism is avoidance wearing a different mask. The paragraph does not need to be perfect right now. It needs to exist. You said you are someone who shows up and does the work. Moving from thinking to doing, even briefly, is how that becomes real.",
         "Leave the paragraph as it is. Move to the next one. You can come back later.", 2),

        # Week 3 - more ups and downs but trending better
        (-16, "08:45", "Woke up early and actually feel good. Going to try to get some work done before class.", "calm", "action", -2.0,
         "You are starting the day with intention. That is not nothing. You said you are someone who shows up and does the work. This steadiness is part of that.",
         "Start with the task that feels smallest. Build from there.", 10),
        (-16, "17:30", "Got loads done today. Finished a whole section of the report and fixed two bugs. I actually feel proud of myself for the first time in weeks.", "proud", "completion", -4.0,
         "You did something hard today. Let that land. Do not rush past this feeling. You said you are someone who shows up and does the work, even when it feels hard. What you accomplished is proof of that.",
         "Rest. Recovery is part of the process. You earned this.", 0),
        (-15, "10:00", "Back to feeling overwhelmed today. I looked at how much is left and it just hit me like a wall. Yesterday felt great but today I am right back to panicking.", "overwhelmed", "overwhelm", 3.0,
         "Yesterday you proved you can do this. Today your brain is scared again. Both things are true. You do not need to solve everything today. You said you are someone who shows up and does the work. Choosing just one thing right now is how that starts.",
         "Pick the one smallest thing on your list. Just that one thing.", 2),
        (-14, "13:00", "I feel bad because I said I would study every day and I have not touched my notes in three days. I keep breaking promises to myself.", "guilty", "avoidance", 3.5,
         "Breaking a streak does not erase what you already built. Every time you showed up before still counts. You said you are someone who studies a little every day. One forward step right now is how that starts again.",
         "Open your notes for 5 minutes. That is enough to restart the streak.", 5),
        (-13, "09:30", "Feeling stressed about the presentation next week. I do not feel prepared at all and the thought of standing up in front of people makes my stomach turn.", "anxious", "overwhelm", 3.0,
         "Anxiety about the presentation means you want to do well. That caring is a strength even when it does not feel like one. You said you are someone who shows up and does the work, even when it feels hard. One small action from here is how that becomes real.",
         "Write down 3 bullet points of what you want to say. Just bullet points, not a script.", 5),
        (-12, "14:30", "Did a 20 minute walk today before sitting down to work. Actually felt clearer after. Maybe there is something to this exercise thing.", "calm", "action", -2.5,
         "You noticed how movement changes your headspace. That is self-awareness in action. You said you are someone who moves their body because it helps them think clearly. What you are doing right now is proof of that.",
         "Keep this connection between movement and clarity. Try it again tomorrow.", 5),

        # Week 4 - building momentum, more positive entries
        (-11, "10:00", "Good morning. I have a clear plan for today and I actually feel ready to start. Made a cup of tea and sitting at my desk with my laptop open.", "calm", "action", -3.0,
         "You created the conditions for starting. Tea, desk, plan. That is not small. That is you designing your environment for success. You said you are someone who shows up and does the work. This steadiness is part of that.",
         "Start with the first thing on your plan. You are already in the right place.", 10),
        (-10, "16:00", "Managed to study for 45 minutes today without getting distracted. Put my phone in another room and it actually helped. Small win but it feels good.", "proud", "action", -3.5,
         "45 minutes of focused work is not a small win. That is real evidence that you can do this. You said you are someone who studies a little every day instead of waiting until the last minute. What you are doing right now is proof of that.",
         "Note what worked (phone in another room) and use it again tomorrow.", 2),
        (-9, "11:30", "Trying to write the testing section of my report and I have no idea what to put. Feel like I am making it up as I go.", "stuck", "rumination", 1.5,
         "First drafts are supposed to feel rough. Nobody writes a perfect section on the first try. You said you are someone who shows up and does the work, even when it feels hard. Moving from thinking to doing, even briefly, is how that becomes real.",
         "Write a messy bullet point list of what you tested. You can turn it into proper sentences later.", 5),
        (-8, "09:00", "I woke up and immediately felt that dread in my stomach about the deadline. It is getting closer and I am not sure I will finish in time.", "anxious", "overwhelm", 2.5,
         "That dread is your brain's alarm system, not a prediction of failure. You have been showing up consistently. The work is getting done even if it does not feel like enough. You said you are someone who shows up and does the work, even when it feels hard. One small action from here is how that becomes real.",
         "List what is actually left. The real list, not the imagined one. It is probably shorter than it feels.", 5),
        (-8, "15:30", "Made the list of what is left. It is actually not as bad as I thought. I think if I do a bit each day I can finish. Feeling a bit calmer now.", "hopeful", "action", -2.0,
         "You just proved something important. The feeling of too much is almost always bigger than the reality. Making the list gave your prefrontal cortex something concrete to work with. You said you are someone who shows up and does the work. This feeling is evidence of that.",
         "Pick one thing from the list and start it now while you have this clarity.", 10),
        (-7, "10:15", "Did not sleep well but I am going to try to work anyway. Even if it is just 30 minutes. I know from experience now that starting is the hardest part.", "tired", "action", -0.5,
         "You know the pattern now. That is awareness. And you are choosing to show up anyway. You said you are someone who shows up and does the work, even when it feels hard. Showing up even now is part of that.",
         "Do 20 minutes of work then take a break. Gentle pace.", 20),

        # Recent days - clear growth visible
        (-5, "09:00", "Good morning. Sat down with my coffee and opened the report. Going to work through the methodology section today. I have a plan and I am sticking to it.", "calm", "action", -3.0,
         "You are showing up with intention again. Coffee, report, plan. This is becoming your routine. You said you are someone who shows up and does the work, even when it feels hard. This steadiness is part of that.",
         "Start the methodology section. You know what to write.", 15),
        (-5, "14:00", "Finished the methodology section. It is rough but it exists. I can edit it later. Right now I am just glad it is done.", "proud", "completion", -4.5,
         "You finished something real. And you already know it does not need to be perfect right now. That is growth. You said you are someone who shows up and does the work. What you just finished is evidence of that.",
         "Take a real break. Walk, stretch, anything away from the screen.", 0),
        (-4, "10:30", "Feeling a bit stressed about the demo but I have been practicing. I know my project well because I built it. That has to count for something.", "stressed", "action", 0.0,
         "You built this. You understand it. The stress is your brain preparing, not warning you of failure. You said you are someone who shows up and does the work, even when it feels hard. One rough action right now is how that starts.",
         "Practice the demo one more time. Focus on the story, not the features.", 10),
        (-3, "11:00", "Had a moment of panic this morning when I realised there are only two weeks left. But then I looked at my list and crossed off what I have already done. I have actually done a lot.", "anxious", "action", 0.5,
         "You interrupted the panic with evidence. That is exactly what awareness looks like. You said you are someone who shows up and does the work, even when it feels hard. One small action from here is how that becomes real.",
         "Pick the next uncrossed item on your list", 10),
        (-3, "17:00", "I cannot believe I actually feel calm about this project for the first time in weeks. I am not done yet but I can see the finish line. Every day I show up it gets a little closer.", "calm", "action", -3.5,
         "You can feel the shift. This is what showing up consistently builds. Not motivation, not confidence, just a quiet knowing that you can do this because you have been doing it. This steadiness is part of that.",
         "Keep going at this pace. You do not need to speed up.", 10),
        (-2, "09:30", "Coded for 2 hours straight this morning. Fixed the last bug on the analytics page and everything works now. I feel genuinely proud.", "proud", "completion", -5.0,
         "Two hours of deep work and a bug squashed. That is not just progress, that is evidence of who you have become during this project. You said you are someone who shows up and does the work. What you just finished is evidence of that.",
         "Rest. You did deep work today and you deserve recovery.", 0),
        (-1, "10:00", "Starting the day with journaling again. I have been doing this almost every day for three weeks now and I can actually feel the difference. I am still anxious sometimes but I know how to sit with it now.", "hopeful", "action", -3.0,
         "Three weeks of showing up. That is not a streak, that is a new pattern. The anxiety does not disappear but your relationship with it has changed. You said you are someone who shows up and does the work, even when it feels hard. This feeling is evidence of that.",
         "Use this awareness today. You know what to do next.", 10),
        (-1, "16:00", "Finished editing two chapters of the report. They are not perfect but they are solid. I used to think everything had to be perfect before I could move on. Now I just keep going.", "calm", "completion", -4.0,
         "That shift from perfectionism to progress is one of the biggest changes you have made. Solid is enough. Moving forward matters more than polishing. This steadiness is part of that.",
         "Take the evening off. Tomorrow is a new day.", 0),
        (0, "09:15", "Morning. Another day, another chance to show up. I have a list and I know what to do. Feeling okay about it.", "calm", "action", -2.5,
         "You are starting the day from a place of steadiness. That used to be rare for you. Notice how different this feels from four weeks ago. You said you are someone who shows up and does the work. This steadiness is part of that.",
         "Start with the first item on your list. You have got this.", 10),
        (0, "13:30", "Working through the testing section. It is challenging but in a good way. I actually enjoy writing about what I built. Three weeks ago I could not even open the laptop.", "hopeful", "action", -3.0,
         "You are enjoying the work. That is not motivation, that is identity alignment. You built something you are proud of and now writing about it feels natural. You said you are someone who shows up and does the work. This feeling is evidence of that.",
         "Keep going. This flow is earned, not random.", 15),

        # a few extra entries on other emotional days to fill out analytics
        (-26, "20:00", "Cannot sleep because I keep thinking about everything I have not done. My mind will not shut off.", "anxious", "rumination", 3.5,
         "Your mind is trying to solve problems while your body needs rest. You do not have to fix everything tonight. You said you are someone who shows up and does the work. One small action from here is how that becomes real.",
         "Write down the 3 things bothering you most. Getting them out of your head onto paper can help your brain let go.", 3),
        (-22, "20:30", "Feeling guilty that I spent the afternoon playing games instead of working. I always do this when things get hard.", "guilty", "avoidance", 4.0,
         "Playing games when things get hard is your brain looking for safety. It is not laziness. Noticing the pattern is how you begin to change it. You said you are someone who shows up and does the work, even when it feels hard. One forward step right now is how that starts again.",
         "Just open your project for 5 minutes before bed. Even glancing at it counts.", 5),
        (-17, "21:00", "I feel a bit better tonight. I actually got some work done and went for a walk. Two things I said I would do and I did them. That counts for something.", "hopeful", "completion", -2.5,
         "You set intentions and followed through. That is identity alignment in action. You said you are someone who shows up and does the work. This feeling is evidence of that.",
         "Let yourself rest now. Tomorrow you will show up again.", 0),
        (-11, "21:30", "Ended the day feeling okay. Not great, not terrible. I did some work, I went for a walk, I ate properly. Normal day. That is probably fine.", "calm", "recovery", -2.0,
         "A normal day where you showed up is a good day. Not every day needs to be a breakthrough. You said you are someone who shows up and does the work. Resting so you can return stronger is part of that.",
         "Sleep well. You did enough today.", 0),
        (-6, "09:30", "I feel like I am behind where I should be. Everyone else seems further along with their projects. Comparison is killing me right now.", "stressed", "overwhelm", 2.0,
         "Comparison tells you nothing about your own progress. You cannot see what everyone else is struggling with behind the scenes. You said you are someone who shows up and does the work, even when it feels hard. One rough action right now is how that starts.",
         "Close social media and open your own project. Your timeline is yours.", 10),
        (-4, "21:00", "Long day but a good one. I am starting to trust that if I just show up and do a little each day it will get done. I do not need to be perfect. I just need to be consistent.", "calm", "recovery", -3.0,
         "That trust is hard-won. It came from weeks of showing up even when you did not want to. You said you are someone who shows up and does the work, even when it feels hard. Resting so you can return stronger is part of that.",
         "Sleep. Consistency includes rest.", 0),
    ]

    entry_ids = []
    for days_offset, time_str, text, emotion, behaviour, p_score, reframe, micro_task, micro_mins in journal_entries:
        d = today + timedelta(days=days_offset)
        created_at = f"{d.isoformat()}T{time_str}:00"

        eid = insert_and_get_id(
            cur,
            "INSERT INTO journal_entries "
            "(user_id, entry_text, predicted_emotion, predicted_behaviour, "
            "paralysis_score, reframe, micro_task_text, micro_task_minutes, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (user_id, text, emotion, behaviour, p_score, reframe, micro_task, micro_mins, created_at)
        )
        entry_ids.append(eid)

    # ---- 11. Todos (mix of manual and journal-sourced, some done, some not) ----
    todo_data = [
        # manual todos
        ("Buy groceries for the week", "manual", None, (today - timedelta(days=20)).isoformat(), 1, (today - timedelta(days=20)).isoformat() + "T10:00:00"),
        ("Email supervisor about demo date", "manual", None, (today - timedelta(days=18)).isoformat(), 1, (today - timedelta(days=18)).isoformat() + "T09:00:00"),
        ("Print research papers for reading", "manual", None, (today - timedelta(days=15)).isoformat(), 1, (today - timedelta(days=15)).isoformat() + "T11:00:00"),
        ("Back up project to USB", "manual", None, (today - timedelta(days=10)).isoformat(), 1, (today - timedelta(days=10)).isoformat() + "T14:00:00"),
        ("Submit ethics form", "manual", None, (today - timedelta(days=7)).isoformat(), 1, (today - timedelta(days=7)).isoformat() + "T09:00:00"),
        ("Fix CSS issue on mobile view", "manual", None, (today - timedelta(days=3)).isoformat(), 1, (today - timedelta(days=3)).isoformat() + "T15:00:00"),
        ("Write test plan document", "manual", None, (today - timedelta(days=1)).isoformat(), 0, (today - timedelta(days=1)).isoformat() + "T10:00:00"),
        ("Proofread chapter 3", "manual", None, today.isoformat(), 0, today.isoformat() + "T08:00:00"),
        ("Add citations to methodology", "manual", None, today.isoformat(), 0, today.isoformat() + "T08:30:00"),
        ("Practice demo with timer", "manual", None, (today + timedelta(days=1)).isoformat(), 0, today.isoformat() + "T09:00:00"),
        # journal-sourced todos (from micro-tasks added to today)
        ("Open your project folder and look at one file for 30 seconds", "journal", entry_ids[0], (today - timedelta(days=28)).isoformat(), 1, (today - timedelta(days=28)).isoformat() + "T10:00:00"),
        ("Write down the one smallest part of what is worrying you", "journal", entry_ids[2], (today - timedelta(days=27)).isoformat(), 1, (today - timedelta(days=27)).isoformat() + "T11:00:00"),
        ("Open your notes for 5 minutes", "journal", entry_ids[17], (today - timedelta(days=14)).isoformat(), 1, (today - timedelta(days=14)).isoformat() + "T14:00:00"),
        ("Pick one thing from the list and start it now", "journal", entry_ids[25], (today - timedelta(days=8)).isoformat(), 1, (today - timedelta(days=8)).isoformat() + "T16:00:00"),
        ("Start with the first item on your list", "journal", entry_ids[35], today.isoformat(), 0, today.isoformat() + "T09:30:00"),
    ]

    for text, source, j_id, due, done, created in todo_data:
        insert_and_get_id(
            cur,
            "INSERT INTO todos (user_id, text, source, journal_entry_id, due_date, is_done, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (user_id, text, source, j_id, due, done, created)
        )

    conn.commit()
    conn.close()

    # summary
    print(f"\nTest user created successfully!")
    print(f"  Username: {TEST_USERNAME}")
    print(f"  Password: {TEST_PASSWORD}")
    print(f"  User ID: {user_id}")
    print(f"  Goals: {len(goal_data)}")
    print(f"  Identity beliefs: {len(belief_data)}")
    print(f"  Positive thoughts: {len(thought_data)}")
    print(f"  Habits: {len(sample_habits) + len(custom_habits)} ({len(sample_habits)} sample + {len(custom_habits)} custom)")
    print(f"  Goal steps: {len(goal_steps_data)}")
    print(f"  Journal entries: {len(journal_entries)}")
    print(f"  Todos: {len(todo_data)}")
    print(f"  Alignment score: 87")
    print(f"  Emotional streak: 5 days")
    print(f"\nData spans {28} days of usage showing a journey from")
    print(f"high paralysis to growing confidence.")
    print(f"\nLog in at /login with the credentials above.")


if __name__ == "__main__":
    main()
