from string_utils.manipulation import roman_encode

def test__roman_encode_mutant_killing():
    """
    Test the roman encoding for values that include tens place to check that the mutant fails.
    The test case uses the number 10 (should return 'X'), which will trigger a failure in the mutant due to the mapping change.
    """
    # This input is a clear detection for the mutant
    input_value = 10
    expected_output = 'X'
    
    output = roman_encode(input_value)
    print(f"Input: {input_value}, Output: {output}")
    assert output == expected_output  # This should pass for baseline but fail for mutant