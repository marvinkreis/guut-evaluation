from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only_re():
    # Test input that contains HTML tags
    test_input = "<html><body></body></html>"
    
    # The regex should successfully match the input HTML and return matches
    matches = HTML_TAG_ONLY_RE.findall(test_input)
    assert len(matches) > 0, "The regex did not match expected HTML input."

    # Test input that contains incorrect HTML tags
    incorrect_test_input = "<some>text</some>"
    
    # This should also match valid HTML tags
    incorrect_matches = HTML_TAG_ONLY_RE.findall(incorrect_test_input)
    assert len(incorrect_matches) > 0, "The regex did not match expected HTML input."

    # Now let's use an input that contains only trivial whitespace or invalid HTML to test the mutant
    mutant_test_input = "This is just some text without tags."
    
    # The regex should not find any matches in this case
    mutant_matches = HTML_TAG_ONLY_RE.findall(mutant_test_input)
    assert len(mutant_matches) == 0, "The regex matched when it shouldn't have, indicating an issue with the mutant."
