from string_utils.manipulation import roman_encode

def test_roman_encode():
    # Known good outputs for basic cases
    test_cases = {
        1: 'I',
        2: 'II',
        3: 'III',
        4: 'IV',
        5: 'V',
        6: 'VI',
        7: 'VII',
        8: 'VIII',
        9: 'IX',
        10: 'X',
        11: 'XI',
        12: 'XII',
        13: 'XIII',
        14: 'XIV',
        15: 'XV',
        16: 'XVI',
        17: 'XVII',
        18: 'XVIII',
        19: 'XIX',
        20: 'XX',
        30: 'XXX',
        31: 'XXXI',
        32: 'XXXII',
        33: 'XXXIII',
        34: 'XXXIV',
        35: 'XXXV',
        36: 'XXXVI',
        36: 'XXXVI',
        38: 'XXXVIII',   
        39: 'XXXIX',
        40: 'XL',
        41: 'XLI',
        50: 'L',
        60: 'LX',
        70: 'LXX',
        80: 'LXXX',
        90: 'XC',
        100: 'C',
        400: 'CD',
        500: 'D',
        900: 'CM',
        1000: 'M'
    }

    # Checking assertions over existing ranges to gather behavior and inconsistencies
    for input_value, expected_output in test_cases.items():
        assert roman_encode(input_value) == expected_output, f"Failed for {input_value}"

    # Invalid input checks for both versions
    for invalid_input in [0, -1, 4000]:
        try:
            roman_encode(invalid_input)
            assert False, f"Expected ValueError for invalid input {invalid_input}, but it did not raise"
        except ValueError:
            pass  # Expected behavior