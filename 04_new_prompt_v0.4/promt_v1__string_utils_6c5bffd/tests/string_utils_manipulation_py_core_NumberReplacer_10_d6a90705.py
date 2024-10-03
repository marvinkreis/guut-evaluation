from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test the encoding of numbers in the hundreds range, particularly 400 and 500.
    The input values 400 and 500 should yield different Roman numeral outputs on the mutant:
    The correct output for 400 is 'CD' and for 500 is 'D'.
    The mutant will fail for 400 due to the incorrect mapping of 'D'.
    """
    output_400 = roman_encode(400)
    output_500 = roman_encode(500)
    
    print(f"Output for 400: {output_400}")
    print(f"Output for 500: {output_500}")
    
    assert output_400 == 'CD'
    assert output_500 == 'D'