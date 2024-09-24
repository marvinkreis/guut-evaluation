from string_utils._regex import PRETTIFY_RE

def test__prettify_re():
    # Test string that includes multiple spaces (should be replaced if regex is correct)
    test_string = "This  is a   test string with   multiple spaces."
    
    # Apply the regex
    result = PRETTIFY_RE['DUPLICATES'].sub(" ", test_string)
    
    # Expected result should collapse multiple spaces into a single space
    expected_result = "This is a test string with multiple spaces."
    
    # Assert that the result matches the expected result
    assert result == expected_result, f"Expected '{expected_result}', but got '{result}'"