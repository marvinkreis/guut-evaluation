from string_utils.generation import roman_range

def test__roman_range():
    """Changing the comparison from '>' to '>=' in roman_range causes valid configurations to raise an error."""
    # This should return ['IV', 'V'] in the correct implementation
    output = list(roman_range(5, start=4, step=1))
    assert output == ['IV', 'V'], "Expected output for range from 4 to 5 should include 'IV' and 'V'."