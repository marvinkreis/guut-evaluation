from string_utils.generation import roman_range

def test_roman_range():
    # Test case where we expect an OverflowError due to invalid start, stop, step configuration
    try:
        # This should raise an OverflowError since we're trying to go backward from 1 to 5 with a positive step
        list(roman_range(stop=5, start=1, step=-1))
        assert False, "Expected an OverflowError but none was raised"
    except OverflowError:
        pass  # Correct behavior, the mutant should fail this case

    # Now let's verify the case with a correct configuration
    # This should generate: IV, V (with a step of 1 from start=4 to stop=5)
    result = list(roman_range(stop=6, start=4, step=1))
    assert result == ['IV', 'V', 'VI'], f"Expected ['IV', 'V', 'VI'] but got {result}"