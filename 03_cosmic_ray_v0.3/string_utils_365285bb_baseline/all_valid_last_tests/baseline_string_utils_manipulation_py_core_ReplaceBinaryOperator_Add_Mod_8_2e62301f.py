import re
from string_utils.manipulation import __StringFormatter

def test__ensure_left_space_only():
    # Create an instance of __StringFormatter
    formatter = __StringFormatter("   this is a test string   ")

    # Simulate a regex match object. We can use the `re` module to create a match.
    match = re.match(r'^(.*)', "this is a test string")
    
    # Now call the __ensure_left_space_only method with the simulated match object
    result = formatter._StringFormatter__ensure_left_space_only(match)

    # The expected behavior: should return a string with a leading space
    expected_output = " this is a test string"  # ' this is a test string'

    # Assert the result
    assert result == expected_output, f"Expected '{expected_output}', but got '{result}'"