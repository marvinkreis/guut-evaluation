from string_utils.generation import random_string

def test_random_string():
    # Test case to check the behavior of random_string with size exactly 1
    output = random_string(1)  # should generate a single random character
    assert len(output) == 1  # Check that the output has length 1
    assert output.isalnum()  # Check that the output is alphanumeric

    # Now we check if the function raises an error when size is less than 1
    try:
        random_string(0)
    except ValueError as e:
        assert str(e) == 'size must be >= 1'  # Check the error message

    try:
        random_string(-5)
    except ValueError as e:
        assert str(e) == 'size must be >= 1'  # Check the error message