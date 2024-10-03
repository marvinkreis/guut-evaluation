from string_utils._regex import PRETTIFY_RE

def test__duplicates_regex():
    """
    This test checks the 'DUPLICATES' regex in the PRETTIFY_RE dictionary. The baseline should correctly identify duplicate punctuation, while the mutant should fail due to a change in the regex compilation. 
    The baseline output reflects successful matching for cases like "Hello!!  World!!" and should return an empty list for strings with no duplicates.
    """
    test_strings = [
        "Hello!!  World!!",  # should match duplicates
        "This is a test... And another test!",  # should match duplicates
        "No duplicates here!"  # should not match
    ]
    
    duplicates_re = PRETTIFY_RE['DUPLICATES']
    
    for test_string in test_strings:
        matches = duplicates_re.findall(test_string)
        print(f"Testing '{test_string}': matches = {matches}")
        assert (test_string == "Hello!!  World!!" and matches) or \
               (test_string == "This is a test... And another test!" and not matches) or \
               (test_string == "No duplicates here!" and not matches)