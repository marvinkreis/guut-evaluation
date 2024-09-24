from string_utils.generation import random_string

def test__random_string():
    # Test with valid size
    result = random_string(5)
    assert len(result) == 5  # The result should have a length of 5

    # Test with invalid size (not an integer)
    try:
        random_string('a')  # Should raise ValueError
        assert False, "Expected ValueError for string input."
    except ValueError:
        pass  # Expected behavior

    # Test with invalid size (negative integer)
    try:
        random_string(-1)  # Should raise ValueError
        assert False, "Expected ValueError for negative size."
    except ValueError:
        pass  # Expected behavior

    # Test with invalid size (zero)
    try:
        random_string(0)  # Should raise ValueError
        assert False, "Expected ValueError for size zero."
    except ValueError:
        pass  # Expected behavior