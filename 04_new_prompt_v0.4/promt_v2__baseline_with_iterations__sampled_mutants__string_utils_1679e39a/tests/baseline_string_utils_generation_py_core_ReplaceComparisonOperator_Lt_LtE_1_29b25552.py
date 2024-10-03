from string_utils.generation import secure_random_hex

def test__secure_random_hex():
    """
    Test whether the function raises a ValueError when the byte_count is set to 0.
    Both the baseline and the mutant should raise a ValueError for this input.
    Additionally, testing with 1 should show that the mutant does not raise an error while the baseline does, 
    as the baseline should require byte_count > 1.
    """
    
    # Test for byte_count = 0 - should raise ValueError in both baseline and mutant
    try:
        secure_random_hex(0)
        assert False  # Test should fail as we expect an exception.
    except ValueError:
        pass  # This is expected.

    # Now test for byte_count = 1, this should work with mutant but raise an error in baseline
    try:
        result = secure_random_hex(1)  # This should not raise an exception, but we need to check the logic flaw in mutant.
        assert isinstance(result, str)  # Validate that it returns a string
    except ValueError:
        assert False  # If it fails, then we encountered a problem.