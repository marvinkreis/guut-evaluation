from string_utils.generation import secure_random_hex

def test_secure_random_hex_mutant_killing():
    """
    Test the secure_random_hex function with a string input.
    The baseline will raise a ValueError due to invalid input,
    while the mutant should raise a TypeError as it incorrectly checks input types.
    """
    try:
        secure_random_hex("five")
    except ValueError:
        # This is the expected behavior for the baseline
        assert True  # The test passes here
    except TypeError:
        # This is the expected behavior for the mutant, indicating it has failed
        assert False, "Mutant should not raise a TypeError for input 'five', but it did."
    else:
        # If we reach this point, no errors were raised, which is unexpected behavior
        assert False, "Expected ValueError or TypeError, but no exception was raised."