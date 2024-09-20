from string_utils.generation import roman_range

def test__roman_range():
    """The mutant modifies the backward-exceed condition leading to an error for valid parameters."""
    
    # Expected output
    expected_output = ['V', 'IV', 'III', 'II', 'I']
    
    # Call the function with a test case
    output = list(roman_range(1, start=5, step=-1))
    
    # Assert that the output matches the expected Roman numeral sequence
    assert output == expected_output, "roman_range must produce the correct sequence of Roman numerals"