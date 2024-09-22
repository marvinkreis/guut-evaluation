from string_utils.generation import roman_range

def test_roman_range():
    # Test to check normal generation of Roman numerals.
    romans = list(roman_range(5))  # Generating Roman numerals from 1 to 5
    assert romans == ['I', 'II', 'III', 'IV', 'V'], f"Expected ['I', 'II', 'III', 'IV', 'V'], got {romans}"

    # Test with valid configuration
    output = list(roman_range(stop=7, start=1, step=1))  # Should generate I to VII
    assert output == ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'], f"Expected ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'], got {output}"

    # This should raise an OverflowError due to invalid parameters (start > stop)
    try:
        list(roman_range(stop=1, start=5, step=1))  # Should raise OverflowError
        assert False, "Expected OverflowError not raised for invalid step configuration, but it was."
    except OverflowError:
        pass  # This is expected behavior

    # Check valid backward stepping
    romans_backward = list(roman_range(start=5, stop=1, step=-1))  
    assert romans_backward == ['V', 'IV', 'III', 'II', 'I'], f"Expected ['V', 'IV', 'III', 'II', 'I'], got {romans_backward}"

    # Invalid backward range that should raise an OverflowError
    try:
        list(roman_range(stop=4, start=1, step=-1))  # Should raise OverflowError
        assert False, "Expected OverflowError not raised for invalid backward exceed, but it was."
    except OverflowError:
        pass  # This is expected

    # Specific case to identify mutant behavior
    try:
        list(roman_range(stop=10, start=1, step=10))  # Large step that should cause overflow
        assert False, "Expected OverflowError not raised for large step with forward exceed, but it was."
    except OverflowError:
        pass  # Expected due to overflow
    
    # Edge case for maximum valid bounds on Roman numerals
    try:
        output = list(roman_range(stop=3999, start=1, step=1))  # Should work well
        assert output[-1] == 'MMMCMXCIX', f"Expected last output to be 'MMMCMXCIX', got {output[-1]}"
    except Exception as e:
        assert False, f"Unexpected error for valid range: {str(e)}"