from string_utils.manipulation import prettify

def test__prettify_saxon_genitive():
    # When the input includes a valid Saxon genitive, the output should preserve it correctly
    input_string = "Dave' s dog"
    expected_output = "Dave's dog"
    
    actual_output = prettify(input_string)
    
    # This assertion should pass for the original function
    assert actual_output == expected_output, f"Expected: '{expected_output}', got: '{actual_output}'"
    
    # After executing this test with the mutant, the output will change incorrectly,
    # which will help in detecting the mutant due to the manipulation of group references.