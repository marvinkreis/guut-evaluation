from string_utils._regex import HTML_TAG_ONLY_RE

def test__DOCTYPE_HTML():
    """
    Test the HTML_TAG_ONLY_RE regex with a DOCTYPE declaration that serves 
    as an edge case to demonstrate the difference in handling between 
    the baseline and the mutant.
    """
    
    input_value = "<!DOCTYPE html><html></html>"
    output = HTML_TAG_ONLY_RE.match(input_value)
    
    # The baseline should match, while the mutant should not.
    assert output is not None, "Expected a match for the baseline."
    print(f"Input: {input_value}, Output: {output}")

# Call the test function (it's just for context here and won't be executed since the execution is handled by your system)