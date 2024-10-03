from string_utils.generation import roman_range

def test__roman_range_backward_steps_invalid():
    """
    Test the roman_range function with parameters designed to yield an invalid configuration.
    The baseline should raise an OverflowError while the mutant raises a ValueError when stepping backward
    with the invalid configuration.
    """
    try:
        list(roman_range(3, start=1, step=-1))  # This should cause issues in both, but with differing exceptions.
    except OverflowError:
        print("Baseline correctly raised OverflowError.")
    except ValueError as e:
        print(f"Mutant raised ValueError: {e}")
        assert False, "Mutant behavior is incorrect, should have raised OverflowError."
    else:
        assert False, "No error raised - should have raised OverflowError."