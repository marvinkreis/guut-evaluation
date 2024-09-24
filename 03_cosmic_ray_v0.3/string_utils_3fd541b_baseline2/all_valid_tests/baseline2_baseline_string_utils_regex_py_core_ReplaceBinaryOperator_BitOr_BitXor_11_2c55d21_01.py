from string_utils._regex import PRETTIFY_RE

def test__PRETTIFY_RE():
    # Testing a string with quoted text to see if the regex matches correctly
    test_string = '"Hello world" and (some text)'
    
    # The original regex matches the quoted text inside double quotes
    assert PRETTIFY_RE['SPACES_INSIDE'].search(test_string) is not None, "Original regex should find quoted text."
    
    # Testing a string without quotes
    test_string_no_quotes = 'Just some plain text.'
    
    # Again, it should return no matches for text without quotes
    assert PRETTIFY_RE['SPACES_INSIDE'].search(test_string_no_quotes) is None, "Original regex should not match text without quotes."
    
    # The mutant regex would incorrectly behave here due to the combination change of MULTILINE and DOTALL.
    # We expect it to work correctly with the original but potentially fail with the mutant.

# This test case should pass with the original code but fail with the mutant due to the change from `|` to `^`.