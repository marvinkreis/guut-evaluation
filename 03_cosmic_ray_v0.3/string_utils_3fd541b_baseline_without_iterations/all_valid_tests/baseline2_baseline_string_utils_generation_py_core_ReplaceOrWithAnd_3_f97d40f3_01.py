from string_utils.generation import roman_range

def test_roman_range_invalid_step_configuration():
    try:
        # This should raise an OverflowError because start (5) is greater than stop (1) with a positive step (1).
        list(roman_range(stop=1, start=5, step=1))
        assert False, "Expected OverflowError was not raised."
    except OverflowError:
        pass  # Test passes
        
    try:
        # This should raise an OverflowError because start (5) is greater than stop (3) with a positive step (2).
        list(roman_range(stop=3, start=5, step=2))
        assert False, "Expected OverflowError was not raised."
    except OverflowError:
        pass  # Test passes
