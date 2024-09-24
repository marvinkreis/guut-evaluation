from string_utils.generation import random_string

def test_random_string():
    # Test with valid input
    result = random_string(10)
    assert len(result) == 10  # The result should be of size 10

    # Test with invalid input (size < 1)
    try:
        random_string(0)
    except ValueError as e:
        assert str(e) == 'size must be >= 1'
    
    # Test with invalid input (size negative)
    try:
        random_string(-5)
    except ValueError as e:
        assert str(e) == 'size must be >= 1'
    
    # Test with size = 1 (to detect the mutant)
    result = random_string(1)
    assert len(result) == 1  # The result should be of size 1

    # Test with size = 2 (to confirm it generates different results)
    result1 = random_string(2)
    result2 = random_string(2)
    assert len(result1) == 2 and len(result2) == 2  # Both results should be of size 2
    assert result1 != result2  # The two results should not be the same (most likely)

# When using the mutant, the assertion for size >= 1 will not raise an error for size = 1, 
# thus passing the test incorrectly.