from string_utils.generation import roman_range

def test__roman_range_valid_mutant_behavior():
    """
    Test the roman_range function with valid parameters for the mutant.
    It should pass and generate Roman numerals in the mutant while 
    raising an exception in the baseline case.
    """
    try:
        # This should work in mutant but raise exception in baseline
        output = list(roman_range(stop=0, start=1))
        print(f"Mutant output: {output}")  # Expecting valid output in mutant
        assert len(output) > 0, "Output should have generated Roman numerals."
    except ValueError as e:
        print(f"Baseline raised ValueError: {e}")
    except OverflowError as e:
        assert False, f"Unexpected OverflowError in mutant: {e}"