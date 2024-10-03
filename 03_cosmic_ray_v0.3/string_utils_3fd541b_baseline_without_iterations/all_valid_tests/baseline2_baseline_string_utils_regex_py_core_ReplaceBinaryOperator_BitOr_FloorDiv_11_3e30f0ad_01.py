from string_utils._regex import PRETTIFY_RE

def test__prettify_re():
    # Test case input
    test_string = 'This is a test string with multiple spaces,          and quotes: "hello world".'
    
    # The expected output is a simplified version where multiple spaces are reduced to single spaces
    expected_output = 'This is a test string with multiple spaces, and quotes: "hello world".'
    
    # Simulate the regex behavior
    # The Python regex compile and match methods are directly not needed here
    # We will check if the pattern matches the expected output after simplification
    cleaned_string = PRETTIFY_RE['DUPLICATES'].sub(' ', test_string)
    
    assert cleaned_string == expected_output, f"Expected: {expected_output}, but got: {cleaned_string}"