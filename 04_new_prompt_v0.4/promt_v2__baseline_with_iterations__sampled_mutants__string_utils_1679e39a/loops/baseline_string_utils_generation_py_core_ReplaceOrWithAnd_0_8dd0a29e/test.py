from string_utils.generation import random_string

def test__random_string():
    """
    Test to ensure that the random_string function raises a ValueError when a negative size is provided.
    This checks the mutant that changes the condition in line 53. The mutant will not raise an error for 
    negative sizes, while the original function should raise a ValueError for any size < 1.
    """
    try:
        random_string(-5)
    except ValueError as e:
        assert str(e) == 'size must be >= 1'
        return
    raise AssertionError("ValueError was not raised for negative size")