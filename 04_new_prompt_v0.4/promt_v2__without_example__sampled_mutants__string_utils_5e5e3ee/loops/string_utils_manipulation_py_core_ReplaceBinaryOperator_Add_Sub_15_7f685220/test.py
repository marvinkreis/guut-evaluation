from string_utils.manipulation import slugify

def test__slugify():
    """
    Test slugify function to verify that it produces the expected slug output.
    The test is designed to fail when running the mutant due to a type error from an incorrect operator.
    """
    input_string = 'Top 10 Reasons To Love Dogs!!!'
    expected_output = 'top-10-reasons-to-love-dogs'
    
    output = slugify(input_string)
    
    assert output == expected_output, f"Expected: {expected_output}, but got: {output}"