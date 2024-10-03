from string_utils.generation import roman_range

def test__roman_range_forward_exceed():
    # This test should raise an OverflowError on both the original and mutant
    # when trying to generate from 5 to 1 with a positive step.
    
    try:
        # This configuration is invalid because we cannot count up from 5 to 1
        list(roman_range(start=5, stop=1, step=1))
        # If we reach this line, the test should fail since we expect an OverflowError
        assert False, "Expected OverflowError but none was raised"
    except OverflowError:
        # This is expected; test passes
        pass
    except Exception as e:
        # If a different exception is raised, this is unexpected
        assert False, f"Unexpected exception raised: {type(e).__name__} - {str(e)}"