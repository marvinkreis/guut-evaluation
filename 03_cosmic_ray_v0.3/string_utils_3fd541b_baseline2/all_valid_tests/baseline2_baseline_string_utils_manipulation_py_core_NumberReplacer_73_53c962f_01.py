from string_utils.manipulation import prettify

def test__prettify():
    input_string = "This  is a test   string."
    expected_output = "This is a test string."
    
    # This will pass with the correct implementation
    assert prettify(input_string) == expected_output
    
    # Additionally testing for a case that should not pass with the mutant
    mutant_input_string = "This  is  a  test  string."  # This will yield the same output with the mutant due to the change
    assert prettify(mutant_input_string) == expected_output  # should fail with the mutant