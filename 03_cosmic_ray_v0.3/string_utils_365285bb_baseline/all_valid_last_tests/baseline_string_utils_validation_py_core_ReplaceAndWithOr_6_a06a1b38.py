from string_utils.validation import is_json

def test_is_json():
    # Test 1: Valid JSON object
    valid_json_string = '{"name": "Peter"}'
    assert is_json(valid_json_string) == True

    # Test 2: Invalid JSON (missing quotes on key)
    invalid_json_string = '{nope: true}'
    assert is_json(invalid_json_string) == False

    # Test 3: Empty string should return False
    empty_string = ''
    assert is_json(empty_string) == False

    # Test 4: Valid JSON with leading and trailing spaces
    spaced_json = '   { "age": 30 }   '
    assert is_json(spaced_json) == True

    # Test 5: Non-full string (whitespace only) should return False
    non_full_string = '   '  
    assert is_json(non_full_string) == False

    # Test 6: Invalid JSON object with a trailing comma
    trailing_comma_string = '{"key": "value",}'
    assert is_json(trailing_comma_string) == False 

    # Test 7: Invalid JSON array structure
    invalid_array = '[1, 2, 3,]'
    assert is_json(invalid_array) == False

    # Test 8: A valid JSON array
    valid_json_array = '["apple", "banana"]'
    assert is_json(valid_json_array) == True 

    # Test 9: Complex valid JSON object
    complex_json_string = '{"name": "John", "age": 30, "cars": ["Ford", "BMW", "Fiat"]}'
    assert is_json(complex_json_string) == True 

    # Test 10: Invalid JSON due to unquoted key
    invalid_structure_string = '{"key": "value" "key2": "value2"}'
    assert is_json(invalid_structure_string) == False 

    # Test 11: Invalid JSON with unexpected characters
    invalid_special_string = '{"name": "Dan", $: "value"}'
    assert is_json(invalid_special_string) == False  

    # Test 12: Completely invalid JSON that should return false
    completely_invalid_string = "{this cannot be json}"
    assert is_json(completely_invalid_string) == False 

    # Test 13: Random text that cannot be JSON
    just_a_text = "random text"
    assert is_json(just_a_text) == False 

    # Test 14: Invalid key type (numeric key not in quotes)
    invalid_combined_json = '{"valid": "data", 42: "invalid"}'
    assert is_json(invalid_combined_json) == False 

    # Test 15: Malformed nested JSON structure
    malformed_nested_json = '{"person": {name: "Alice", age: 25}}'
    assert is_json(malformed_nested_json) == False  # Invalid due to unquoted keys in nested object

    # Test 16: valid nested structure
    valid_nested_json = '{"employee": {"name": "Alice", "age": 25}}'
    assert is_json(valid_nested_json) == True  # Should return True

    # Test 17: Well-formed but empty JSON object
    empty_json_object = '{}'
    assert is_json(empty_json_object) == True  # Still valid JSON

# This collection of tests should thoroughly examine conditions affected by the mutant's logical change.