from string_utils.generation import random_string

def test__random_string_mutant_killing():
    """
    This test case is designed to kill the mutant by checking for 
    a size of 5. The baseline is expected to return a valid random
    string of length 5, while the mutant should raise a ValueError 
    as it only accepts a size of exactly 1.
    """
    # Testing with size 5
    try:
        output = random_string(5)
        print(f"output: {output}")  # This should print a valid random string
        assert isinstance(output, str) and len(output) == 5
    except ValueError as e:
        print(f"ValueError: {e}")
        assert False  # Ensure that we fail the test if an exception is raised