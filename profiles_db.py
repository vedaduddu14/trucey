import os
import sqlite3

print("Creating profiles_db.py")

conn = sqlite3.connect('profiles.db')
conn.row_factory = sqlite3.Row  # This makes results work like dictionaries

conn.execute("""CREATE TABLE IF NOT EXISTS profiles(LoginID text, AssignedSystem text, AssignedProblem text, PersonofInterest text, Topic text, RelationshipQuality text, RelationshipLength text, PaymentType text, PreviousInteraction text)""")

print("Profiles table created!")

print("Adding test profiles...")

conn.execute("""INSERT INTO profiles (LoginID, AssignedSystem, AssignedProblem, PersonofInterest, Topic, RelationshipQuality, RelationshipLength, PaymentType, PreviousInteraction) 
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
             ('profile_001', 'trucey_rehearsal', 'asking_for_raise', 'boss', 'salary', 'good', '2_years', 'hourly', 'no_previous_interaction'))
conn.execute("""INSERT INTO profiles (LoginID, AssignedSystem, AssignedProblem, PersonofInterest, Topic, RelationshipQuality, RelationshipLength, PaymentType, PreviousInteraction)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              ('profile_002', 'control_system', 'asking_for_promotion', 'boss', 'salary', 'bad', '5_years', 'salary', 'no_previous_interaction'))
conn.execute("""INSERT INTO profiles (LoginID, AssignedSystem, AssignedProblem, PersonofInterest, Topic, RelationshipQuality, RelationshipLength, PaymentType, PreviousInteraction)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              ('profile_003', 'trucey_rehearsal', 'time_off', 'boss', 'salary', 'neutral', '2_months', 'stipend', 'previously_discussed'))
conn.commit()
print("Test profiles added!")

print("🔧 Adding assignment system columns...")

conn.execute("ALTER TABLE profiles ADD COLUMN prolific_id TEXT")
conn.execute("ALTER TABLE profiles ADD COLUMN status TEXT DEFAULT 'available'")
conn.execute("UPDATE profiles SET status = 'available' WHERE status IS NULL")

conn.commit()
print("Assignment system ready!")

def assign_profile_to_prolific_id(prolific_id):
    existing = conn.execute("SELECT AssignedSystem, AssignedProblem, PersonofInterest, Topic, RelationshipQuality, RelationshipLength, FROM profiles WHERE prolific_id = ?", (prolific_id,)).fetchone()
    if existing:
        return dict(existing)
    available = conn.execute("SELECT * FROM profiles WHERE status = 'available' LIMIT 1").fetchone()
    if available:
        conn.execute("UPDATE profiles SET prolific_id = ?, status = 'assigned' WHERE LoginID = ?", 
                    (prolific_id, available['LoginID']))
        conn.commit()
        return {
            'AssignedSystem': available['AssignedSystem'],
            'AssignedProblem': available['AssignedProblem'],
            'PersonofInterest': available['PersonofInterest'],
            'Topic': available['Topic'],
            'RelationshipQuality': available['RelationshipQuality'],
            'RelationshipLength': available['RelationshipLength'],
            'prolific_id': prolific_id
        }
    else:
        return None


conn.close()