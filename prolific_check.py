def check_prolific_data(prolific_id):
    """Quick check of what data a prolific ID has"""
    from revised_backend_CLEAN import authenticate_participant, convert_sqlite_to_info_format
    
    print(f"Checking prolific_id: '{prolific_id}'")
    
    # Get raw data
    data = authenticate_participant(prolific_id)
    if not data:
        print("❌ No data found for this prolific_id")
        return None
    
    print("✅ Raw data found:")
    for key, value in data.items():
        print(f"  {key}: {value}")
    
    # Get converted info
    info = convert_sqlite_to_info_format(data)
    print(f"\n✅ Scenario info:")
    print(f"  Topic: {info['topic']['value']} ({info['topic']['description']})")
    print(f"  Person: {info['individual']['value']} ({info['individual']['description']})")
    print(f"  System: {data['assigned_system']}")
    
    return data, info

# Quick database query function
def check_database_for_prolific(prolific_id):
    """Direct database query"""
    import sqlite3
    
    conn = sqlite3.connect('./profiles.db')
    conn.row_factory = sqlite3.Row
    
    result = conn.execute("""
        SELECT * FROM profiles WHERE prolific_id = ?
    """, (prolific_id,)).fetchone()
    
    if result:
        print(f"✅ Found in database:")
        for key in result.keys():
            print(f"  {key}: {result[key]}")
    else:
        print(f"❌ prolific_id '{prolific_id}' not found in database")
    
    conn.close()
    return dict(result) if result else None

# One-liner for quick testing
def quick_check(prolific_id):
    """Super quick check"""
    from revised_backend_CLEAN import authenticate_participant
    data = authenticate_participant(prolific_id)
    if data:
        print(f"✅ {prolific_id}: {data['assigned_system']} | {data['assigned_problem']} | {data['person_of_interest']}")
    else:
        print(f"❌ {prolific_id}: Not found")
    return data

# Usage examples:
if __name__ == "__main__":
    # Test whatever prolific_id you want
    test_id = "prolific123"  # Change this
    
    print("=== QUICK CHECK ===")
    quick_check(test_id)
    
    print("\n=== DETAILED CHECK ===")
    check_prolific_data(test_id)
    
    print("\n=== DATABASE CHECK ===")
    check_database_for_prolific(test_id)