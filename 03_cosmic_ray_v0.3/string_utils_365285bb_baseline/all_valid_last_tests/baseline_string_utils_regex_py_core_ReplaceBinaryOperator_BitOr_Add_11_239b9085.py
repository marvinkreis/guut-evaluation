from string_utils._regex import PRETTIFY_RE

def test_PRETTIFY_RE():
    # Test input to capture quotes and brackets, handling newlines and extra spaces
    test_string = '"First Line"\n\n"Second Line" and it is (inside brackets)'
    
    # Find matches using the regex
    matches = PRETTIFY_RE['SPACES_INSIDE'].findall(test_string)

    # Expected matches excluding new lines
    expected_matches = ['First Line', 'Second Line', 'inside brackets']
    
    # Clean matches by filtering out any empty results
    filtered_matches = [m.strip() for m in matches if m.strip()]

    assert filtered_matches == expected_matches, f"Expected matches don't match. Found: {filtered_matches}"

    # Test with no quotes or brackets
    test_no_quotes = 'A simple line with nothing special.'
    matches_no_quotes = PRETTIFY_RE['SPACES_INSIDE'].findall(test_no_quotes)

    assert matches_no_quotes == [], f"Expected no matches for test_no_quotes. Found: {matches_no_quotes}"

    # String with valid and empty captures in brackets
    test_with_empty_brackets = 'Here we have (text) but also ( ) empty.'
    matches_empty_brackets = PRETTIFY_RE['SPACES_INSIDE'].findall(test_with_empty_brackets)

    # Expecting valid content but not including empty captures
    expected_empty_capture = ['text']
    filtered_empty_brackets = [m.strip() for m in matches_empty_brackets if m.strip()]

    assert filtered_empty_brackets == expected_empty_capture, f"Expected valid bracket text only. Found: {filtered_empty_brackets}"

    # Test for leading/trailing spaces
    test_spaces = '   "   Leading quote"   and (this is it)   '
    matches_spaces = PRETTIFY_RE['SPACES_INSIDE'].findall(test_spaces)

    expected_space_capture = ['Leading quote', 'this is it']
    
    filtered_spaces = [m.strip() for m in matches_spaces if m.strip()]

    assert filtered_spaces == expected_space_capture, f"Expected matches with spaces don't match. Found: {filtered_spaces}"

    # Test multiline handling: Should properly capture quoted lines while locking out newlines
    test_multiline = '"Alpha Line"\n\n"Beta Line" and it is (included text)  '
    matches_multiline = PRETTIFY_RE['SPACES_INSIDE'].findall(test_multiline)

    expected_multiline = ['Alpha Line', 'Beta Line', 'included text']
    
    filtered_multiline = [m.strip() for m in matches_multiline if m.strip()]

    assert filtered_multiline == expected_multiline, f"Expected matches with multiline don't match. Found: {filtered_multiline}"

    # Finally, check only newlines yielding zero matches
    test_only_newlines = '\n\n\n'
    matches_only_newlines = PRETTIFY_RE['SPACES_INSIDE'].findall(test_only_newlines)

    assert matches_only_newlines == [], f"Expected only newlines to yield empty. Found: {matches_only_newlines}"

# Execute the test function
test_PRETTIFY_RE()