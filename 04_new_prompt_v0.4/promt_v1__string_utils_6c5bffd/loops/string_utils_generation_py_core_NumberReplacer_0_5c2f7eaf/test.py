from string_utils.generation import random_string

def test__random_string_kills_mutant():
    """
    Test whether random_string can return a single character when size = 1.
    The Baseline implementation should return one character, while the Mutant
    should raise a ValueError due to its altered input validation.
    """
    try:
        output = random_string(1)
        assert output is not None and len(output) == 1  # Ensure output is a single character
    except ValueError:
        assert False  # This should not happen in Baseline