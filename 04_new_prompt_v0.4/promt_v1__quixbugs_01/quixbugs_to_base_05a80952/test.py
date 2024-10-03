from to_base import to_base

def test__to_base_kill_mutant():
    """
    Test the to_base function with input values that will result in different outputs for the baseline and the mutant.
    The input value is 31 and the base is 16. The expected output for the baseline is '1F', but for the mutant,
    it will produce 'F1' due to the order of concatenation being altered.
    """
    output = to_base(31, 16)
    assert output == '1F'  # This should pass for the baseline but fail for the mutant.