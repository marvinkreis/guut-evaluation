from string_utils.generation import secure_random_hex

def test__secure_random_hex_with_assertion():
    """
    Test the secure_random_hex function with a non-integer input.
    The expected behavior for the baseline includes raising a ValueError,
    while the mutant raises a TypeError due to its incorrect condition.
    This test will pass on the baseline and fail on the mutant.
    """
    try:
        secure_random_hex("5")
        assert False, "Expected ValueError was not raised."
    except ValueError:
        print("Caught expected ValueError in baseline.")
    except TypeError as e:
        assert False, f"Mutant raised TypeError: {e}"