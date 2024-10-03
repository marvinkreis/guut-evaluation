from string_utils.generation import random_string

def test__random_string_invalid_size():
    """
    Test that passing a non-integer value or a size less than 1 raises a ValueError.
    The mutant alters the validation logic, which might lead to unexpected behavior
    instead of raising the appropriate ValueError.
    """
    
    # Test with a float, expecting ValueError in Baseline - should fail in Mutant
    try:
        random_string(9.5)
        assert False, "Expected ValueError wasn't raised for non-integer size."
    except ValueError:
        pass  # Correctly handled in Baseline

    # Test with a size of 0, expecting ValueError
    try:
        random_string(0)
        assert False, "Expected ValueError wasn't raised for size 0."
    except ValueError:
        pass  # Correctly handled in Baseline

    # Test with a negative size, expecting ValueError
    try:
        random_string(-5)
        assert False, "Expected ValueError wasn't raised for negative size."
    except ValueError:
        pass  # Correctly handled in Baseline