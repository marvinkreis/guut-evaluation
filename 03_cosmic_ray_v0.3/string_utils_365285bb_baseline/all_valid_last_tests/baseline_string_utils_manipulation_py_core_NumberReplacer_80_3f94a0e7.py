from string_utils.manipulation import prettify

def test_prettify_saxon_genitive():
    input_string = "Dave' s dog"  # Correct Saxon genitive structure
    expected_output = "Dave's dog"  # Expected output after prettification
    actual_output = prettify(input_string)
    
    # Assertion to check if the output is correct
    assert actual_output == expected_output, f"Expected: '{expected_output}', but got: '{actual_output}'"