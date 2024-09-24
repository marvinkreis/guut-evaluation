from string_utils.validation import is_isbn_10

def test_is_isbn_10():
    # Valid ISBN-10 numbers
    assert is_isbn_10('0306406152') == True, 'Test Case 1 Failed'  # Valid ISBN-10 - Expected True
    assert is_isbn_10('1506715214') == True, 'Test Case 2 Failed'  # Valid ISBN-10 - Expected True

    # Edge cases to check between valid and mutant
    # A valid checksum calculation for ISBN-10 to yield False in a mutant scenario
    assert is_isbn_10('1234567890') == False, 'Test Case 3 Failed'  # This should yield False correctly
    assert is_isbn_10('1234567891') == False, 'Test Case 4 Failed'  # Known invalid ISBN-10 return False

    # Test some edge invalid but nearby numbers
    assert is_isbn_10('1234567892') == False, 'Test Case 5 Failed'  # Also Invalid, should yield False under both
    
    # Constructed test for clarity on boundaries
    assert is_isbn_10('0471958697') == True, 'Test Case 6 Failed'  # Another valid ISBN-10
    
    # Input specifically designed for checking mutant behavior
    assert is_isbn_10('0471958691') == False, 'Test Case 7 Failed'  # Should yield False under true logic
    
    print('All test cases passed!')

test_is_isbn_10()