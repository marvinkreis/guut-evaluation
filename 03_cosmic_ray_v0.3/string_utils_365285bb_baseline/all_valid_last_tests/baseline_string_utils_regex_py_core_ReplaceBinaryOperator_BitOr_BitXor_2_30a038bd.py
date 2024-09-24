from string_utils._regex import WORDS_COUNT_RE

def test__words_count():
    # Input string with clearly defined words
    test_string = "Café naïve résumé Hello world"

    # We expect to find the following 5 words: Café, naïve, résumé, Hello, world
    expected_found_words = ['Café', 'naïve', 'résumé', 'Hello', 'world']

    # Use the regex to find words
    found_words = WORDS_COUNT_RE.findall(test_string)

    # Now we should get the accurate number of words found
    # Strip any trailing spaces (this should not be an issue in well-formed input)
    found_words = [word.strip() for word in found_words]

    # Assert the number of words found matches the expected count
    assert len(found_words) == len(expected_found_words), f"Expected {len(expected_found_words)} words, found: {len(found_words)}"

    # Assert that the content of the found words matches the expected output
    assert found_words == expected_found_words, f"Expected words: {expected_found_words}, found: {found_words}"

# Usually, you would call the test function in your main execution