from string_utils.generation import roman_range

def test__roman_range_zero_step():
    """
    Test the roman_range function with a step value of zero.
    A ValueError should be raised in the baseline while the mutant leads to a timeout
    indicating incorrect handling of zero step.
    """
    try:
        output = list(roman_range(stop=10, step=0))
        print(f"output = {output}")
    except ValueError as e:
        print(f"ValueError raised as expected: {e}")
    except Exception as e:
        print(f"Unexpected exception raised: {e}")