from string_utils.generation import roman_range

def test_roman_range_mutant_detection():
    # Testing the original implementation of roman_range
    expected_output = ["V", "IV", "III", "II", "I"]
    result = list(roman_range(start=5, stop=1, step=-1))
    
    # Check if the result matches the expected output
    assert result == expected_output, f"Expected {expected_output}, but got {result}."
    
    # If the mutant is active, it will likely fail to compute the correct roman numerals,
    # and raise an OverflowError or produce an incorrect sequence.