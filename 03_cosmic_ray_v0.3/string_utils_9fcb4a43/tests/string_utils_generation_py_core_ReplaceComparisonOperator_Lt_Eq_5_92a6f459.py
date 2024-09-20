from string_utils.generation import roman_range

def test__roman_range_edge_cases():
    """The mutant allows invalid range configurations that the correct version does not."""
    
    # Test for the case where start equals stop with a negative step
    try:
        output = list(roman_range(5, 5, -1))
        assert False, f"Expected OverflowError, but got output: {output}"
    except OverflowError as e:
        assert str(e) == 'Invalid start/stop/step configuration'

    # Additionally check the behavior of start equals stop with a positive step
    try:
        output = list(roman_range(5, 5, 1))
        assert False, f"Expected OverflowError, but got output: {output}"
    except OverflowError as e:
        assert str(e) == 'Invalid start/stop/step configuration'