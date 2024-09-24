from string_utils.generation import roman_range

def test_roman_range_invalid_case():
    # Invalid configuration where start < stop and step is negative should raise an OverflowError
    try:
        generator = roman_range(start=2, stop=5, step=-1)  # Invalid backwards attempt
        result = list(generator)
        assert False, "Expected OverflowError was not raised."
    except OverflowError:
        # Expected behavior for the original function
        pass

def test_roman_range_invalid_case_reverse():
    # Invalid configuration where step is positive but start > stop should raise OverflowError
    try:
        generator = roman_range(start=5, stop=2, step=1)  # Invalid forward attempt
        result = list(generator)
        assert False, "Expected OverflowError was not raised."
    except OverflowError:
        # Expected behavior for the original function
        pass

def test_roman_range_valid_case():
    # Valid scenario: going from 3 to 1 with -1 step should return ['III', 'II', 'I']
    generator = roman_range(start=3, stop=1, step=-1)
    result = list(generator)
    assert result == ['III', 'II', 'I'], f"Expected ['III', 'II', 'I'] but got {result}"

# Execute the tests
test_roman_range_invalid_case()
test_roman_range_invalid_case_reverse()
test_roman_range_valid_case()