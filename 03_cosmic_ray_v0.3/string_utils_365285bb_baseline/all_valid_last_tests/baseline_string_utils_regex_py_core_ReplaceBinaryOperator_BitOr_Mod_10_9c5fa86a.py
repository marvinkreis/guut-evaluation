from string_utils._regex import PRETTIFY_RE
import re

def test_prettify_re():
    # Test for multiple spaces between words
    test_string = "This  is   a test    string."
    expected_output = "This is a test string."  # Spaces should be reduced to a single space
    
    # Use the PRETTIFY_RE pattern to normalize spaces
    result = re.sub(PRETTIFY_RE['DUPLICATES'], ' ', test_string)

    # Check if the output matches the expected result
    assert result == expected_output, f"Expected: '{expected_output}', but got: '{result}'"

    # Test for leading/trailing spaces
    test_string_with_spaces = "   Excessive spaces    before and after.    "
    expected_after_trimming = "Excessive spaces before and after."
    
    # Normalizing and trimming spaces using regex
    trimmed_result = re.sub(PRETTIFY_RE['DUPLICATES'], ' ', test_string_with_spaces).strip()

    # Check for expected result
    assert trimmed_result == expected_after_trimming, f"Expected: '{expected_after_trimming}', but got: '{trimmed_result}'"

    # Test for quoted text with excessive spaces
    test_quote = "    \"Hello   world!\"    "
    expected_quote_output = "\"Hello world!\""
    
    # Normalize internal spaces and trim:
    quote_normalized = re.sub(PRETTIFY_RE['DUPLICATES'], ' ', test_quote.strip())
    
    # Check if the result matches expected format
    assert quote_normalized == expected_quote_output, f"Expected: '{expected_quote_output}', but got: '{quote_normalized}'"