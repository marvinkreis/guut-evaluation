from string_utils._regex import HTML_RE

def test__html_regex_unordered_markup():
    """Test HTML_RE against unordered and incorrect markup to expose potential mutant behavior."""

    # Unordered HTML-like input with multiple closure mismatches
    unordered_markup = "<div><span>Inline <p>Malformed </span><p></div><div>Extra</div>"

    # Testing the correct implementation
    match_builtin = HTML_RE.match(unordered_markup)
    if match_builtin:
        print("Matched groups in correct implementation:", match_builtin.groups())
    else:
        print("Correct implementation did not match unexpectedly.")

    # Testing the mutant implementation
    mutant_match = HTML_RE.match(unordered_markup)
    if mutant_match:
        print("Matched groups in mutant implementation:", mutant_match.groups())
    else:
        print("Mutant implementation did not match unexpectedly.")