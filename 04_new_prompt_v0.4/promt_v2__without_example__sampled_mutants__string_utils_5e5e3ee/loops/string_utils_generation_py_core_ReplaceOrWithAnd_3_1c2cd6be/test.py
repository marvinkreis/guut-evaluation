from string_utils.generation import roman_range

def test__roman_range_mutant_killer():
    """
    Test the behavior of the roman_range function when start equals stop
    and step is positive. The baseline raises an OverflowError, while the mutant
    returns the Roman numeral 'V' for the number 5. This test confirms the 
    difference in their behaviors.
    """
    try:
        result = list(roman_range(start=5, stop=5, step=1))
        output = ' '.join(result)  # This will be 'V' for the mutant
    except OverflowError:
        output = "OverflowError"
    
    assert output == "OverflowError", f"Expected 'OverflowError', but got {output}"