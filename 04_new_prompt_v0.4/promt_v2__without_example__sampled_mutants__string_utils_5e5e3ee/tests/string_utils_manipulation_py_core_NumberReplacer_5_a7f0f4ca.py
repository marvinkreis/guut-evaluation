from string_utils.manipulation import roman_encode

def test__roman_encode_tens():
    """
    Test the encoding process for numbers 10, 11, and 12 where the baseline should return correct Roman numerals:
    - 10 => "X"
    - 11 => "XI"
    - 12 => "XII"

    The mutant should fail to return these correct outputs due to the KeyError.
    """
    # Baseline expected outputs
    expected_outputs = {
        10: 'X',
        11: 'XI',
        12: 'XII'
    }

    for input_number, expected_output in expected_outputs.items():
        output = roman_encode(input_number)
        print(f"Input: {input_number}, Output: {output}")

        # Assert correct results expected from the baseline
        assert output == expected_output, f"Expected {expected_output} for input {input_number}, got {output}"

    # The mutant will trigger a KeyError on 10 when attempting to fetch a mapping that doesn't exist