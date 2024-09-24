from string_utils.generation import roman_range

def test_roman_range():
    # Testing for a valid range with forward step
    result = list(roman_range(7, start=1, step=1))  # Expected: ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
    assert result == ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'], f"Expected 'I' to 'VII', got {result}"

    # Testing for a range that should raise OverflowError due to invalid configuration
    try:
        list(roman_range(1, start=5, step=1))  # Invalid as it can't step down to 1
        assert False, "Expected OverflowError but did not get it"
    except OverflowError:
        pass  # Expected behavior

    # This will check the behavior of the mutant
    # Forward step exceeding test that will fail under the mutant condition
    try:
        list(roman_range(5, start=10, step=1))  # This should raise OverflowError
        assert False, "Expected OverflowError but did not get it"
    except OverflowError:
        pass  # Expected behavior

# Run the test
test_roman_range()