from to_base import to_base

def test__to_base():
    """The mutant's change to the concatenation order causes the outputs to be reversed for multi-digit numbers."""
    
    # Testing the case where the mutation manifests:
    assert to_base(31, 16) == '1F', "Incorrect conversion from base 10 to base 16"
    assert to_base(10, 2) == '1010', "Incorrect conversion from base 10 to base 2"