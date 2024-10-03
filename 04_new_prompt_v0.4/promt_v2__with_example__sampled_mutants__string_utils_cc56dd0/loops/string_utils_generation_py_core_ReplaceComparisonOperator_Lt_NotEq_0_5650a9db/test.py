from string_utils.generation import random_string

def test_random_string_mutant_killing():
    """
    Test the random_string function using a size of 2. The mutant will raise a ValueError,
    while the baseline will return a string of length 2.
    """
    try:
        output = random_string(2)
        assert len(output) == 2, f"Expected length 2, got {len(output)}"
    except ValueError as e:
        assert False, f"Unexpected ValueError: {e}"