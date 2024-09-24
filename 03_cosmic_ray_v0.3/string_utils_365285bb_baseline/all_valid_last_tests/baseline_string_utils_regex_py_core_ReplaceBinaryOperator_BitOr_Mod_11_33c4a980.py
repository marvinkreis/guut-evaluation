from string_utils._regex import PRETTIFY_RE

def test_prettify_re():
    # Test input string for the original operation
    test_string = '"Hello!" and (some data)'

    # The expected output from the original regex functionality
    expected_correct_matches = ['Hello!', 'some data']

    # Obtain matches using the original regex
    correct_matches = PRETTIFY_RE['SPACES_INSIDE'].findall(test_string)

    # Assert that the original matches reflect the expected outputs
    assert correct_matches == expected_correct_matches, \
        f"Expected matches {expected_correct_matches}, but got {correct_matches}"

    # Introduce a specific mutant test string
    mutant_string = '"Potential problem!" and (unexpected data) extra text.'

    # Try to capture matches using the mutant regex behavior
    mutant_matches = PRETTIFY_RE['SPACES_INSIDE'].findall(mutant_string)

    # Define what we might expect under the mutant behavior
    expected_mutant_output = ['Potential problem!', ' and (', 'unexpected data', ' extra text.']

    # Ensure mutant output does not match correct expected output
    assert mutant_matches != expected_correct_matches, \
        f"Mutant should produce different matches. Got {mutant_matches} instead of {expected_correct_matches}"

    # Check to see if output format under mutant is broken due to miscompilation
    assert mutant_matches != expected_mutant_output, \
        f"Expected mutant output to differ, got {mutant_matches}, expected different structure."
