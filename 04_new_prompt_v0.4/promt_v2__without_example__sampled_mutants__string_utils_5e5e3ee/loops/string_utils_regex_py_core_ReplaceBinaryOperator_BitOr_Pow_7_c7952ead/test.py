from string_utils._regex import PRETTIFY_RE

def test__regex_multiline_dotall():
    """
    Test that the PRETTIFY_RE 'DUPLICATES' regex properly handles multiple lines with duplicates.
    The test confirms that the mutant, which altered the regex flags, will throw an error,
    while the baseline will execute successfully and match appropriately.
    """
    test_string = "This is a test string with multiple lines.\n\nIt includes commas, periods, and other punctuation!"
    output = PRETTIFY_RE['DUPLICATES'].findall(test_string)
    assert output == ['\n\n']