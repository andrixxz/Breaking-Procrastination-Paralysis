# setup_rls.py
# turns on row-level security for all data tables in supabase
# run this once after migrate_to_postgres.py has created the tables
#
# how it works:
# - the flask app sets a postgres session variable (app.current_user_id) on each connection
# - RLS policies check that variable to only let users see their own data
# - even if there is a bug in the flask code, the database itself blocks data leakage
#
# users table is excluded because login/signup need to query it
# before we know who the user is

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    print("No DATABASE_URL found in .env file. Cannot set up RLS.")
    exit(1)

print("Connecting to Supabase PostgreSQL...")
conn = psycopg2.connect(DATABASE_URL)
conn.autocommit = True
cur = conn.cursor()

# create a role for the flask app that actually respects RLS
# the default postgres role has BYPASSRLS so policies get ignored without this
# NOLOGIN because we connect as postgres then SET ROLE to this
print("\nSetting up flask_app role...")
cur.execute("SELECT 1 FROM pg_roles WHERE rolname = 'flask_app';")
if cur.fetchone() is None:
    cur.execute("CREATE ROLE flask_app NOLOGIN NOBYPASSRLS;")
    print("  created flask_app role (NOBYPASSRLS)")
else:
    print("  flask_app role already exists")

# give it access to everything it needs
cur.execute("GRANT USAGE ON SCHEMA public TO flask_app;")
cur.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO flask_app;")
cur.execute("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO flask_app;")
cur.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO flask_app;")
cur.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO flask_app;")

# let postgres switch to this role via SET ROLE
cur.execute("GRANT flask_app TO postgres;")
print("  permissions granted, postgres can SET ROLE to flask_app")

# tables that have a user_id column - standard policy
STANDARD_TABLES = [
    'goals',
    'journal_entries',
    'todos',
    'alignment_state',
    'habits',
    'identity_beliefs',
    'positive_thoughts',
    'goal_steps',
    'step_completions',
]

# habit_completions is special - no user_id column, linked through habits table


def drop_existing_policies(table_name):
    """remove any old policies so we can recreate them cleanly"""
    cur.execute("""
        SELECT policyname FROM pg_policies WHERE tablename = %s;
    """, (table_name,))
    existing = cur.fetchall()
    for row in existing:
        policy_name = row[0]
        cur.execute(f'DROP POLICY IF EXISTS "{policy_name}" ON {table_name};')
        print(f"    dropped old policy: {policy_name}")


print("\nSetting up Row-Level Security...\n")

# ------------------------------------------------------------------
# standard tables - all have user_id, same policy pattern
# ------------------------------------------------------------------
for table in STANDARD_TABLES:
    print(f"  {table}:")

    # turn on RLS
    cur.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;")

    # force it to apply even for the postgres superuser role
    # without this, the postgres role bypasses all policies
    cur.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY;")

    # clear out any old policies first
    drop_existing_policies(table)

    # SELECT - only see your own rows
    cur.execute(f"""
        CREATE POLICY "{table}_select" ON {table}
        FOR SELECT
        USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::integer);
    """)

    # INSERT - can only insert rows with your own user_id
    cur.execute(f"""
        CREATE POLICY "{table}_insert" ON {table}
        FOR INSERT
        WITH CHECK (user_id = NULLIF(current_setting('app.current_user_id', true), '')::integer);
    """)

    # UPDATE - can only update your own rows
    cur.execute(f"""
        CREATE POLICY "{table}_update" ON {table}
        FOR UPDATE
        USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::integer);
    """)

    # DELETE - can only delete your own rows
    cur.execute(f"""
        CREATE POLICY "{table}_delete" ON {table}
        FOR DELETE
        USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::integer);
    """)

    print(f"    RLS enabled + 4 policies created (select/insert/update/delete)")


# ------------------------------------------------------------------
# habit_completions - no user_id, ownership via habits.user_id
# ------------------------------------------------------------------
print(f"  habit_completions:")

cur.execute("ALTER TABLE habit_completions ENABLE ROW LEVEL SECURITY;")
cur.execute("ALTER TABLE habit_completions FORCE ROW LEVEL SECURITY;")

drop_existing_policies('habit_completions')

# SELECT - only see completions for your own habits
cur.execute("""
    CREATE POLICY "habit_completions_select" ON habit_completions
    FOR SELECT
    USING (
        habit_id IN (
            SELECT id FROM habits
            WHERE user_id = NULLIF(current_setting('app.current_user_id', true), '')::integer
        )
    );
""")

# INSERT - only insert completions for your own habits
cur.execute("""
    CREATE POLICY "habit_completions_insert" ON habit_completions
    FOR INSERT
    WITH CHECK (
        habit_id IN (
            SELECT id FROM habits
            WHERE user_id = NULLIF(current_setting('app.current_user_id', true), '')::integer
        )
    );
""")

# UPDATE - only update completions for your own habits
cur.execute("""
    CREATE POLICY "habit_completions_update" ON habit_completions
    FOR UPDATE
    USING (
        habit_id IN (
            SELECT id FROM habits
            WHERE user_id = NULLIF(current_setting('app.current_user_id', true), '')::integer
        )
    );
""")

# DELETE - only delete completions for your own habits
cur.execute("""
    CREATE POLICY "habit_completions_delete" ON habit_completions
    FOR DELETE
    USING (
        habit_id IN (
            SELECT id FROM habits
            WHERE user_id = NULLIF(current_setting('app.current_user_id', true), '')::integer
        )
    );
""")

print(f"    RLS enabled + 4 policies created (subquery via habits table)")


# ------------------------------------------------------------------
# users table - special policies
# supabase auto-enabled RLS on this table so we need policies
# SELECT and INSERT are open (login needs to look up any username, signup creates users)
# UPDATE and DELETE restricted to own row (profile edits, GDPR account deletion)
# ------------------------------------------------------------------
print(f"  users:")

# make sure RLS is on (supabase probably already did this)
cur.execute("ALTER TABLE users ENABLE ROW LEVEL SECURITY;")

# force it so flask_app role respects policies
cur.execute("ALTER TABLE users FORCE ROW LEVEL SECURITY;")

drop_existing_policies('users')

# SELECT - login needs to look up any username to check credentials
cur.execute("""
    CREATE POLICY "users_select" ON users
    FOR SELECT
    USING (true);
""")

# INSERT - signup needs to create new users (no user_id set yet)
cur.execute("""
    CREATE POLICY "users_insert" ON users
    FOR INSERT
    WITH CHECK (true);
""")

# UPDATE - only update your own account
cur.execute("""
    CREATE POLICY "users_update" ON users
    FOR UPDATE
    USING (id = NULLIF(current_setting('app.current_user_id', true), '')::integer);
""")

# DELETE - only delete your own account (GDPR right to erasure)
cur.execute("""
    CREATE POLICY "users_delete" ON users
    FOR DELETE
    USING (id = NULLIF(current_setting('app.current_user_id', true), '')::integer);
""")

print(f"    RLS enabled + 4 policies (select/insert open, update/delete own-row only)")


# ------------------------------------------------------------------
# verify everything worked
# ------------------------------------------------------------------
print("\n\nVerifying RLS status...\n")

# pg_tables has rowsecurity but forcerowsecurity is in pg_class
cur.execute("""
    SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public' AND c.relkind = 'r'
    ORDER BY c.relname;
""")

rows = cur.fetchall()
for row in rows:
    table_name = row[0]
    rls_on = row[1]
    force_on = row[2]
    status = "RLS ON" if rls_on else "no RLS"
    force = " + FORCED" if force_on else ""
    print(f"  {table_name}: {status}{force}")

print("\n\nVerifying policies...\n")

cur.execute("""
    SELECT tablename, policyname, cmd
    FROM pg_policies
    WHERE schemaname = 'public'
    ORDER BY tablename, policyname;
""")

policies = cur.fetchall()
for p in policies:
    print(f"  {p[0]}: {p[1]} ({p[2]})")

print(f"\nTotal policies created: {len(policies)}")

cur.close()
conn.close()

print("\nRLS setup complete.")
print("The database now blocks cross-user data access at the DB level.")
print("Even if the flask code has a bug, users can only see their own data.")
