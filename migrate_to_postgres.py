# migrate_to_postgres.py
# creates all tables in supabase postgres
# run this once to set up the production database schema
# tables are created in the right order so foreign keys work

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    print("No DATABASE_URL found in .env file. Cannot migrate.")
    exit(1)

print("Connecting to Supabase PostgreSQL...")
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

# all tables use SERIAL PRIMARY KEY instead of sqlite's INTEGER PRIMARY KEY AUTOINCREMENT
# keeping created_at as TEXT to match what the flask app stores (iso format strings)
# date() function works on these text values in postgres because it casts them to DATE type

print("Creating tables...")

# 1. users - no dependencies
cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        onboarding_complete INTEGER NOT NULL DEFAULT 0
    );
""")
print("  users table ready")

# 2. goals - depends on users
cur.execute("""
    CREATE TABLE IF NOT EXISTS goals (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id),
        goal_text TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
""")
print("  goals table ready")

# 3. journal_entries - depends on users
cur.execute("""
    CREATE TABLE IF NOT EXISTS journal_entries (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id),
        entry_text TEXT NOT NULL,
        predicted_emotion TEXT NOT NULL,
        predicted_behaviour TEXT,
        paralysis_score REAL,
        reframe TEXT,
        micro_task_text TEXT,
        micro_task_minutes INTEGER,
        created_at TEXT NOT NULL
    );
""")
print("  journal_entries table ready")

# 4. todos - depends on users and journal_entries
cur.execute("""
    CREATE TABLE IF NOT EXISTS todos (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id),
        text TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'manual',
        journal_entry_id INTEGER REFERENCES journal_entries(id),
        due_date TEXT,
        is_done INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    );
""")
print("  todos table ready")

# 5. alignment_state - one row per user, tracks score and streak
cur.execute("""
    CREATE TABLE IF NOT EXISTS alignment_state (
        id SERIAL PRIMARY KEY,
        user_id INTEGER UNIQUE NOT NULL REFERENCES users(id),
        alignment_score INTEGER NOT NULL,
        emotional_streak INTEGER NOT NULL,
        last_journal_date TEXT
    );
""")
print("  alignment_state table ready")

# 6. habits - depends on users and goals (linked_goal_id is optional FK)
cur.execute("""
    CREATE TABLE IF NOT EXISTS habits (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id),
        name TEXT NOT NULL,
        is_sample INTEGER NOT NULL DEFAULT 0,
        is_hidden INTEGER DEFAULT 0,
        linked_goal_id INTEGER REFERENCES goals(id),
        created_at TEXT NOT NULL
    );
""")
print("  habits table ready")

# 7. habit_completions - tracks when habits get done
cur.execute("""
    CREATE TABLE IF NOT EXISTS habit_completions (
        id SERIAL PRIMARY KEY,
        habit_id INTEGER NOT NULL REFERENCES habits(id),
        completed_at TEXT NOT NULL
    );
""")
print("  habit_completions table ready")

# 8. identity_beliefs - "I am someone who..." statements from onboarding
cur.execute("""
    CREATE TABLE IF NOT EXISTS identity_beliefs (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id),
        belief_text TEXT NOT NULL,
        linked_goal_id INTEGER REFERENCES goals(id),
        created_at TEXT NOT NULL
    );
""")
print("  identity_beliefs table ready")

# 9. positive_thoughts - linked to identity beliefs
cur.execute("""
    CREATE TABLE IF NOT EXISTS positive_thoughts (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id),
        thought_text TEXT NOT NULL,
        linked_belief_id INTEGER REFERENCES identity_beliefs(id),
        created_at TEXT NOT NULL
    );
""")
print("  positive_thoughts table ready")

# 10. goal_steps - breaking goals into small actionable pieces
cur.execute("""
    CREATE TABLE IF NOT EXISTS goal_steps (
        id SERIAL PRIMARY KEY,
        goal_id INTEGER NOT NULL REFERENCES goals(id),
        user_id INTEGER NOT NULL REFERENCES users(id),
        step_text TEXT NOT NULL,
        step_order INTEGER NOT NULL DEFAULT 1,
        frequency TEXT NOT NULL DEFAULT 'one-off',
        due_date TEXT,
        day_of_week INTEGER,
        is_done INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    );
""")
print("  goal_steps table ready")

# 11. step_completions - per-day tracking for daily/weekly steps
cur.execute("""
    CREATE TABLE IF NOT EXISTS step_completions (
        id SERIAL PRIMARY KEY,
        step_id INTEGER NOT NULL REFERENCES goal_steps(id),
        user_id INTEGER NOT NULL REFERENCES users(id),
        completed_date TEXT NOT NULL
    );
""")
print("  step_completions table ready")

conn.commit()
cur.close()
conn.close()

print("\nAll 11 tables created successfully in Supabase PostgreSQL.")
print("Tables: users, goals, journal_entries, todos, alignment_state,")
print("        habits, habit_completions, identity_beliefs, positive_thoughts,")
print("        goal_steps, step_completions")
