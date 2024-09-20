from string_utils.generation import random_string

def test__random_string():
    """The mutant changes the condition to allow only `size == 1`, which leads to errors for valid sizes (i.e., sizes > 1)."""
    output = random_string(1)
    assert len(output) == 1, "random_string should return a length of 1"