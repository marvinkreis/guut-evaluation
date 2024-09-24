from string_utils.generation import roman_range

def test_roman_range():
    # Test an invalid range where start equals stop with a negative step
    try:
        # This should raise OverflowError for the correct implementation
        result = list(roman_range(5, start=5, step=-1))
        # If no exception is raised, the mutant passes which is not expected
        assert False, "Expected an OverflowError, but no exception was raised."
    except OverflowError:
        # Correct implementation raises OverflowError, test passes
        pass
    except Exception as e:
        # Fail if any other exception is raised
        assert False, f"Expected OverflowError, but got {type(e).__name__}: {e}"

    # Additionally, test valid ascending range
    expected_output = ['I', 'II', 'III', 'IV', 'V']
    assert list(roman_range(5)) == expected_output, "Should return I, II, III, IV, V"