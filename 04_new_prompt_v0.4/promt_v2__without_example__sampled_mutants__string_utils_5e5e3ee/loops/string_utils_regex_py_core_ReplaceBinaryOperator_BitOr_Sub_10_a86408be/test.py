from string_utils._regex import PRETTIFY_RE

def test__prettify_re_duplicates():
    """
    Test the DUPLICATES regex within the PRETTIFY_RE dictionary.
    The input string contains repetitions of characters (like multiple spaces),
    which should match correctly in the baseline but will cause a failure in the mutant
    due to the changes in regex compilation flags leading to an error.
    """
    test_string = "This is a    test string with     duplicates!!!"
    output = PRETTIFY_RE['DUPLICATES'].findall(test_string)
    assert output == ['    ', '     '], f"Expected matches did not occur: {output}"

# Uncomment below to run the test
# test__prettify_re_duplicates()