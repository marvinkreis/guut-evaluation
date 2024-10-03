from string_utils.generation import roman_range

def test_roman_range_mutant_killing():
    """
    Test the roman_range function using parameters that should trigger an OverflowError.
    The baseline will raise OverflowError due to invalid start/stop/step configuration, 
    while the mutant should raise ValueError instead.
    """
    try:
        output = list(roman_range(stop=10, start=1, step=10))
        assert False, "Expected OverflowError, but did not raise"
    except OverflowError as e:
        print(f"Caught expected OverflowError: {e}")
    except Exception as e:
        # The mutant should not raise this exception
        assert False, f"Expected OverflowError, but raised different exception: {e}"