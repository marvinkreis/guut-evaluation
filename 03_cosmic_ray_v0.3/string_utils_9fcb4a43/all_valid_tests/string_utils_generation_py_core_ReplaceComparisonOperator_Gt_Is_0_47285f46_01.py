from string_utils.generation import roman_range

def test__roman_range():
    """This test checks that the roman_range function handles a simple case correctly."""
    output = list(roman_range(10, 1, 1))
    expected_output = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
    assert output == expected_output, "roman_range must correctly generate roman numerals from 1 to 10."
    
    # Testing an invalid range
    try:
        list(roman_range(5, 1, -1))  # This should raise an OverflowError
    except OverflowError:
        pass  # An OverflowError is expected and this test should pass