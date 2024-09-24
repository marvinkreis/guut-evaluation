from string_utils.generation import random_string

def test_random_string():
    # Testing with a size of 1, should return a single character string
    result = random_string(1)
    assert len(result) == 1, "Expected a string of length 1"
    
    # Testing with a size of 5, should return a string of exactly that length
    result = random_string(5)
    assert len(result) == 5, "Expected a string of length 5"
    
    # Testing the case where size is 0, this should raise a ValueError
    try:
        random_string(0)
        assert False, "Expected ValueError for size < 1"
    except ValueError:
        pass  # Expected behavior

    # Testing the case where size is negative, this should also raise a ValueError
    try:
        random_string(-5)
        assert False, "Expected ValueError for size < 1"
    except ValueError:
        pass  # Expected behavior