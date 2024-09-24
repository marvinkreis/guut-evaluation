from string_utils.generation import roman_range

def test_roman_range():
    # Test valid backward iteration
    output = list(roman_range(1, 7, -1))
    expected_output = ['VII', 'VI', 'V', 'IV', 'III', 'II', 'I']
    assert output == expected_output, f"Expected {expected_output}, but got {output}"

    # Test invalid backward iteration (should raise OverflowError)
    try:
        list(roman_range(3, 1, -1))
    except OverflowError:
        pass  # Expected, the test should reach here without any issue
    except Exception as e:
        assert False, f"Expected OverflowError, but got {type(e).__name__}"

    # Test invalid configuration with a mutant (step fails with the mutant)
    try:
        list(roman_range(7, 3, -2))  # This should fail based on original implementation
    except OverflowError:
        pass  # Expected to raise an error, so we catch and pass
    except Exception as e:
        assert False, f"Expected OverflowError, but got {type(e).__name__}"

    # Test valid forward iteration
    output = list(roman_range(7, 1, 1))
    expected_output = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
    assert output == expected_output, f"Expected {expected_output}, but got {output}"