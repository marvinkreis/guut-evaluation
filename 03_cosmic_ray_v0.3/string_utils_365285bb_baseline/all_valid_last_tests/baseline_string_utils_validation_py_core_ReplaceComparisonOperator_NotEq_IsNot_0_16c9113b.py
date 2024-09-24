from string_utils.validation import is_full_string

def test_is_full_string():
    # 1. Test for a string of spaces
    assert is_full_string('   ') == False, "Expected False for string with only spaces"
    
    # 2. Test for a normal string
    assert is_full_string('Hello') == True, "Expected True for a non-empty string"
    
    # 3. Test for empty string
    assert is_full_string('') == False, "Expected False for an empty string"
    
    # 4. Test for None
    assert is_full_string(None) == False, "Expected False for None input"
    
    # 5. Test for string of newlines
    assert is_full_string('\n') == False, "Expected False for string with newlines"
    
    # 6. Test for spaces around a character
    assert is_full_string(' a ') == True, "Expected True for string with leading/trailing spaces and a character"
    
    # 7. Test for string with only tabs
    assert is_full_string('\t') == False, "Expected False for string with only tab characters"
    
    # 8. Test for valid character with spaces
    assert is_full_string('   b  ') == True, "Expected True for string with spaces and a character"
    
    # 9. Test for all spaces specifically targeting mutant logic
    assert is_full_string('     ') == False, "Expected False for string with only spaces."

    # 10. Test a string with valid characters, expected True
    assert is_full_string('   test string   ') == True, "Expected True for string with valid content with spaces."

    # 11. Test for leading/trailing spaces around a valid character
    assert is_full_string('       a       ') == True, "Expected True for string with valid character and spaces."

    # 12. Test for mixed spaces around valid characters
    assert is_full_string('  mixed  space  ') == True, "Expected True for string with mixed spaces surrounding valid characters."

# Call the test function
if __name__ == "__main__":
    test_is_full_string()