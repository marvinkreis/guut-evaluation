from string_utils._regex import HTML_RE

def test__html_regex_extreme_malformed():
    """Test HTML_RE against extremely malformed HTML to identify potential mutant behavior."""
    
    # Highly malformed and contradictory markup
    extreme_malformed_markup = "<div><p>Text with <span>nested <p>tags</span></p></div><div>Extra <p></div>"

    # Testing the correct implementation
    match_correct = HTML_RE.match(extreme_malformed_markup)
    if match_correct:
        print("Matched groups in correct implementation:", match_correct.groups())
    else:
        print("Correct implementation did not match unexpectedly.")
    
    # Testing the mutant implementation
    match_mutant = HTML_RE.match(extreme_malformed_markup)
    if match_mutant:
        print("Matched groups in mutant implementation:", match_mutant.groups())
    else:
        print("Mutant implementation did not match unexpectedly.")