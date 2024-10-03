from string_utils.generation import random_string

def test_random_string_mutant_killing():
    """
    Test the random_string function to ensure the mutant is identified.
    The test confirms that a size of 0 raises a ValueError as expected.
    For size of 1, the baseline should return a valid random string.
    The mutant will fail to produce a valid output.
    """
    # Testing with a size of 0 should raise ValueError in the baseline.
    try:
        random_string(0)
    except ValueError as ve:
        assert str(ve) == 'size must be >= 1', f"Unexpected error message: {ve}"
        print(f"Caught ValueError for size=0 as expected: {ve}")
    else:
        raise AssertionError("Expected ValueError for size=0, but none was raised.")

    # Testing with a size of 1 should return a random character.
    output = random_string(1)
    assert len(output) == 1, f"Expected output length of 1, got length: {len(output)}"
    print(f"Output (size=1): {output}")