from string_utils.generation import random_string

def test__random_string():
    """
    Test the random_string function with a size of 0. The baseline should raise a ValueError as the size is below the minimum,
    while the mutant incorrectly allows a size of 0 and would try to generate a string. This will demonstrate that the mutant
    does not handle the size constraint properly, and thus the test will fail when executed with the mutant.
    """
    try:
        random_string(0)
        assert False  # Should not reach this line as ValueError is expected
    except ValueError:
        assert True  # Expected outcome