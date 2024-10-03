import re
from string_utils.manipulation import __StringFormatter

def test__ensure_spaces_around():
    """
    Test the __ensure_spaces_around method of the StringFormatter class. 
    The input string should return a string with spaces formatted correctly for the baseline.
    The mutant should raise a TypeError due to the incorrect usage of the bitwise AND operator instead of string concatenation.
    """
    formatter = __StringFormatter("example")
    # Simulate a regex match object
    match = re.match(r"(.*)", "test string")
    
    # This should succeed for the baseline
    output = formatter._StringFormatter__ensure_spaces_around(match)
    assert output == " test string ", f"Expected ' test string ', but got {output}"