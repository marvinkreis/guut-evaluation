from to_base import to_base

def test__to_base_kill_mutant():
    """
    Test the base conversion functionality to check if the mutant correctly outputs the representation.
    For num = 31 and base = 16, the expected output is '1F' (Baseline) and should not match 'F1' (Mutant).
    """
    assert to_base(31, 16) == '1F'  # This should pass in baseline but fail in mutant
    assert to_base(255, 16) == 'FF'  # This should pass in both and is just for validation

test__to_base_kill_mutant()