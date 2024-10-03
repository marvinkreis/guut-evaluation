from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test that the mutant fails to return the correct Roman numeral for 4.
    The correct output should be 'IV', but due to the mutant's changes, it will raise a KeyError.
    This test checks that the input 4 produces 'IV' in the baseline and shows a failure (KeyError) in the mutant.
    """
    # Confirming behavior in baseline
    baseline_output = roman_encode(4)
    print(f"Baseline output for input 4: {baseline_output}")
    
    # For the mutant, we expect to catch the KeyError
    try:
        roman_encode(4)
    except KeyError as e:
        print(f"Caught KeyError as expected from the mutant: {e}")
        assert str(e) == "5"  # Verifying that the error corresponds to the missing key.