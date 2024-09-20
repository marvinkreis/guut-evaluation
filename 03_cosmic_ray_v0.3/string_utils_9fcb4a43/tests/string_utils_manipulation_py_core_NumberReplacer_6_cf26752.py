from string_utils.manipulation import roman_encode

def test__roman_encode():
    """Tests to confirm that the mutant's mapping change prevents correct roman numeral encoding."""
    # Define inputs that should trigger mutant's failure
    test_cases = {
        50: 'L',   # Correct code should return 'L'
        60: 'LX',  # Correct code should return 'LX'
        70: 'LXX', # Correct code should return 'LXX'
        80: 'LXXX',# Correct code should return 'LXXX'
        90: 'XC'   # Correct code should return 'XC'
    }
    
    for input_number, expected_output in test_cases.items():
        output = roman_encode(input_number)
        assert output == expected_output, f"Expected {expected_output} but got {output} for input {input_number}."