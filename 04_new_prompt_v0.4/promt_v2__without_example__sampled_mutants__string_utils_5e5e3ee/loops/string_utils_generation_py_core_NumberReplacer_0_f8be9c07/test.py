from string_utils.generation import random_string

def test__random_string_size_one():
    """
    Test whether ValueError is raised for size less than 2. The input of size 1 should 
    pass in the baseline (original code), while the mutant code should raise a 
    ValueError due to changed validation logic.
    """
    try:
        random_string(1)
    except ValueError:
        # If we catch the ValueError, the test should fail in the mutant
        assert False, "ValueError should not have been raised in the baseline"
    
    # If we reach here, the baseline succeeded
    assert True