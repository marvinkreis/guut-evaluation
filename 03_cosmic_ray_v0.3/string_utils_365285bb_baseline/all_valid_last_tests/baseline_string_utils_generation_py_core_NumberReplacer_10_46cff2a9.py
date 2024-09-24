from string_utils.generation import roman_range

def test_roman_range():
    
    # Check the upper limit (3999)
    result = list(roman_range(3999))  # This will test the highest valid entry
    expected_3999 = 'MMMCMXCIX'  # Expected last value for 3999
    assert result[-1] == expected_3999, f"Expected last value for 3999 to be {expected_3999}, got {result[-1]}."

    # Now use a test number just above the upper range to see proper handling
    try:
        list(roman_range(4000))  # This should raise ValueError in the original code
        assert False, "Expected ValueError not raised for out-of-bounds stop value of 4000!"
    except ValueError:
        pass  # This is expected for the original code to validate it correctly handles limits.

    # Test the valid range just above (e.g., 3999)
    try:
        result = list(roman_range(3998, 1, 1))  # Should work and produce values up to 3998
        if len(result) == 0:
            assert False, "Expected results for input 3998 but got no output."
        
        # Let's check the last element for validity
        if 'MMMCMXCVIII' not in result:  # As expected for input 3998
            assert False, "The output for input 3998 did not include expected final result."
            
    except Exception as e:
        assert False, f"Unexpected exception raised for valid input: {e}"

    # Testing with a push of 3999 again but check stopping behavior
    try:
        result = list(roman_range(2000, 1, 1))  # Valid start/stop; should be valid under normal conditions
        assert result[-1] == 'MM'  # Check the end value
    except Exception as e:
        assert False, f"Unexpected exception raised for valid input: {e}"

    # Now test with negative step which should lead to OverflowError
    try:
        list(roman_range(10, 1, -1))  # This should raise OverflowError
        assert False, "Expected OverflowError not raised for invalid step configuration!"
    except OverflowError:
        pass  # This is expected for both codes

    # Check for zero step which should also raise ValueError
    try:
        list(roman_range(5, 1, 0))  # Should raise ValueError
        assert False, "Expected ValueError not raised for zero step!"
    except ValueError:
        pass  # Expected behavior in both codes

# Save this updated test function and run it to verify results.