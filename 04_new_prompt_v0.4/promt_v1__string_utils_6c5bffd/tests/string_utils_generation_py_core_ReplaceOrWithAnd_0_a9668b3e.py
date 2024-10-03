from string_utils.generation import random_string

def test__random_string():
    """
    Test the random_string function for various invalid inputs.
    The baseline should raise ValueError when size is less than 1 or not an integer.
    The mutant may fail to raise ValueError for some inputs due to the mutation in the condition check.
    
    This test checks the behavior of the function with a valid input size and various invalid inputs including 
    a string, negative integers, and zero, which should all raise ValueErrors in the baseline but may not in the mutant.
    """
    
    # Test with a valid integer size
    output_valid = random_string(5)
    assert len(output_valid) == 5  # The output should be a string of length 5
    
    # Test with invalid size (string)
    try:
        random_string("foo")
        assert False, "Expected ValueError for string input, but none was raised."
    except ValueError as e:
        print(f"Caught ValueError as expected with string input: {e}")
    
    # Test with invalid size (negative integer)
    try:
        random_string(-5)
        assert False, "Expected ValueError for negative integer, but none was raised."
    except ValueError as e:
        print(f"Caught ValueError as expected with negative integer: {e}")
        
    # Test with invalid size (zero)
    try:
        random_string(0)
        assert False, "Expected ValueError for size zero, but none was raised."
    except ValueError as e:
        print(f"Caught ValueError as expected with size zero: {e}")